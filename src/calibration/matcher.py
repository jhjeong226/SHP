# src/calibration/matcher.py
"""
DataMatcher  ──  FDR 거리가중 대표값 + CRNP 일자료 매칭
=========================================================

[입력]
  - {station_id}_FDR_daily.xlsx   : 깊이별 시트 (date | E1 | E2 | ... | mean)
  - {station_id}_CRNP_daily.xlsx  : (date | N_corrected | Pa | abs_humidity | ...)

[처리]
  1. FDR 깊이별 시트 로드
  2. reference_depths 선택 (YAML calibration.reference_depths, 디폴트 [10])
  3. 깊이가 여러 개면 단순 평균 → 지점별 단일 θ 생성
  4. Schron (2017) 거리가중 평균으로 사이트 대표 θ_field 산출
     - 거리 정보: YAML sensors[site_id].distance
     - 가중치: W(r) ∝ exp(-r / D86)  (D86: 86% 풋프린트 반경)
     - D86는 당일 기상(Pa, abs_humidity, N_corrected)으로 결정
     - crnpy 사용 불가 시 단순 역거리 가중 fallback
  5. CRNP 일자료의 N_corrected 와 date 기준 inner join

[출력]  DataFrame  columns: date | theta_field | N_corrected
  → StandardCalibrator.calibrate(matched_df) 의 입력
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class DataMatcher:
    """
    Parameters
    ----------
    station_cfg : dict   (YAML station 섹션)
    options     : dict   (YAML processing_options 섹션)
    processed_dir : Path (FDR_daily.xlsx, CRNP_daily.xlsx 가 있는 폴더)
    """

    def __init__(self, station_cfg: Dict, options: Dict, processed_dir: Path):
        self.station_cfg   = station_cfg
        self.options       = options
        self.processed_dir = Path(processed_dir)
        self.station_id    = station_cfg["station_info"]["id"]

        soil = station_cfg.get("soil_properties", {})
        self.bulk_density: float = float(soil.get("bulk_density", 1.44))

        cal = options.get("calibration", {})
        self.ref_depths: List[int] = cal.get("reference_depths", [10])

        # YAML sensors → {site_id: distance_m} 맵
        self.sensor_distances: Dict[str, float] = {
            sid.upper(): float(info.get("distance", 0))
            for sid, info in station_cfg.get("sensors", {}).items()
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 공개 인터페이스
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        매칭 수행.

        Returns
        -------
        DataFrame  columns: date | theta_field | fdr_avg | N_corrected | Pa | abs_humidity
          - theta_field : Schron(2017) 거리가중 평균
          - fdr_avg     : reference_depths 지점 단순 평균 (Desilets_01 방식)
          NaN 행 포함 (날짜는 유지, 값 없으면 NaN)
        """
        print(f"\n{'='*60}")
        print(f"  DataMatcher  ─  {self.station_id}")
        print(f"  reference_depths : {self.ref_depths}")
        print(f"{'='*60}")

        # 1. 파일 로드
        fdr_sheets = self._load_fdr()
        crnp_df    = self._load_crnp()

        # 2. reference_depths 기준으로 지점별 θ 산출
        theta_by_site = self._depth_average(fdr_sheets)
        print(f"\n  활성 지점: {sorted(theta_by_site.columns.drop('date'))}  "
              f"({len(theta_by_site.columns) - 1}개)")

        # 3a. 거리가중 평균 → θ_field (Schron 2017)
        theta_field = self._distance_weighted(theta_by_site, crnp_df)

        # 3b. 단순 평균 → fdr_avg
        site_cols = [c for c in theta_by_site.columns if c != "date"]
        fdr_avg_df = theta_by_site[["date"]].copy()
        fdr_avg_df["fdr_avg"] = theta_by_site[site_cols].mean(axis=1)

        # 4. theta_field + fdr_avg 합치기
        theta_combined = pd.merge(theta_field, fdr_avg_df, on="date", how="outer")

        # 5. CRNP N_corrected 와 join
        matched = self._join_crnp(theta_combined, crnp_df)

        n_valid = matched[["theta_field", "N_corrected"]].notna().all(axis=1).sum()
        print(f"\n  매칭 완료: 전체 {len(matched)}일 / 유효 {n_valid}일")
        print(f"  기간: {matched['date'].iloc[0]} ~ {matched['date'].iloc[-1]}")

        return matched

    # ─────────────────────────────────────────────────────────────────────────
    # 1. 파일 로드
    # ─────────────────────────────────────────────────────────────────────────

    def _load_fdr(self) -> Dict[int, pd.DataFrame]:
        """
        FDR_daily.xlsx 에서 reference_depths 에 해당하는 시트만 로드.
        반환: {depth_cm: DataFrame(date | E1 | E2 | ...)}
        """
        path = self.processed_dir / "fdr" / f"{self.station_id}_FDR_daily.xlsx"
        if not path.exists():
            raise FileNotFoundError(f"FDR daily 파일 없음: {path}")

        sheets: Dict[int, pd.DataFrame] = {}
        for depth in self.ref_depths:
            sheet_name = f"{depth}cm"
            try:
                df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
                df["date"] = pd.to_datetime(df["date"]).dt.date
                # mean 컬럼 제외 (지점별 컬럼만 유지)
                site_cols = [c for c in df.columns if c not in ("date", "mean")]
                sheets[depth] = df[["date"] + site_cols]
                print(f"  ✅ FDR {sheet_name}: {len(df)}일 / 지점 {site_cols}")
            except Exception as e:
                raise ValueError(f"FDR 시트 '{sheet_name}' 읽기 실패: {e}")

        return sheets

    def _load_crnp(self) -> pd.DataFrame:
        """CRNP_daily.xlsx 로드."""
        path = self.processed_dir / "crnp" / f"{self.station_id}_CRNP_daily.xlsx"
        if not path.exists():
            raise FileNotFoundError(f"CRNP daily 파일 없음: {path}")

        df = pd.read_excel(path, engine="openpyxl")
        df["date"] = pd.to_datetime(df["date"]).dt.date

        need = ["N_corrected"]  # N_uts는 있으면 사용, 없어도 무방
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"CRNP 파일에 필수 컬럼 없음: {missing}")

        print(f"  ✅ CRNP daily: {len(df)}일")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 2. 깊이별 평균 → 지점별 단일 θ
    # ─────────────────────────────────────────────────────────────────────────

    def _depth_average(self,
                       sheets: Dict[int, pd.DataFrame]) -> pd.DataFrame:
        """
        reference_depths 가 여러 개면 깊이별 단순 평균.
        단일 깊이면 그대로 반환.
        반환: DataFrame  columns: date | {site_id1} | {site_id2} | ...
        """
        if len(sheets) == 1:
            return list(sheets.values())[0]

        # 공통 date 기준으로 깊이별 평균
        first = list(sheets.values())[0]
        site_cols = [c for c in first.columns if c != "date"]

        stacked = pd.concat(
            [df.set_index("date")[site_cols] for df in sheets.values()],
            axis=0,
        )
        averaged = stacked.groupby(level=0).mean().reset_index()
        averaged.rename(columns={"index": "date"}, inplace=True)
        return averaged

    # ─────────────────────────────────────────────────────────────────────────
    # 3. 거리가중 평균 → θ_field
    # ─────────────────────────────────────────────────────────────────────────

    def _distance_weighted(self,
                           theta_by_site: pd.DataFrame,
                           crnp_df: pd.DataFrame) -> pd.DataFrame:
        """
        Schron (2017) 거리가중 평균.

        가중치 계산:
          W(r) ∝ exp(-r / D86)
          D86 = 풋프린트 반경 [m] — 기압·습도·중성자 수로 결정

        crnpy 사용 가능하면 crnpy.footprint_D86() 활용.
        불가하면 D86 = 300m 고정 후 역거리 지수가중 fallback.

        반환: DataFrame  columns: date | theta_field
        """
        site_cols = [c for c in theta_by_site.columns if c != "date"]

        # 지점별 거리 벡터 (순서: site_cols 와 동일)
        distances = np.array([
            self.sensor_distances.get(s.upper(), 0.0) for s in site_cols
        ], dtype=float)

        # CRNP 일자료 병합 (D86 계산용)
        merged = pd.merge(
            theta_by_site,
            crnp_df[["date", "N_corrected",
                      *[c for c in ["Pa", "abs_humidity"] if c in crnp_df.columns]]],
            on="date", how="left",
        )

        results = []
        for _, row in merged.iterrows():
            theta_vals = row[site_cols].values.astype(float)
            valid_mask = np.isfinite(theta_vals)

            if valid_mask.sum() == 0:
                results.append(np.nan)
                continue

            # D86 결정
            d86 = self._calc_D86(
                N   = row.get("N_corrected", np.nan),
                Pa  = row.get("Pa",          np.nan),
                hum = row.get("abs_humidity",np.nan),
            )

            # 가중치: exp(-r / D86)
            r    = distances[valid_mask]
            w    = np.exp(-r / d86)
            w    = w / w.sum()

            results.append(float(np.sum(w * theta_vals[valid_mask])))

        out = theta_by_site[["date"]].copy()
        out["theta_field"] = results
        return out

    def _calc_D86(self, N: float, Pa: float, hum: float) -> float:
        """
        Schron (2017) 풋프린트 반경 D86 [m].
        crnpy.footprint_D86() 사용, 실패 시 300m 고정.
        """
        try:
            import crnpy
            if all(np.isfinite([N, Pa, hum])):
                d86 = float(crnpy.footprint_D86(
                    N   = N,
                    bulk_density = self.bulk_density,
                    Wlat = 0.0,
                    Wsoc = 0.01,
                ))
                return max(d86, 10.0)   # 최소 10m
        except Exception:
            pass
        return 300.0   # fallback

    # ─────────────────────────────────────────────────────────────────────────
    # 4. CRNP N_corrected 와 join
    # ─────────────────────────────────────────────────────────────────────────

    def _join_crnp(self,
                   theta_combined: pd.DataFrame,
                   crnp_df: pd.DataFrame) -> pd.DataFrame:
        """
        theta_combined (date | theta_field | fdr_avg) 와
        crnp_df        (date | N_corrected | Pa | abs_humidity | ...) 를 join.

        반환: date | theta_field | fdr_avg | N_corrected | Pa | abs_humidity
        """
        keep_cols = ["date", "N_corrected"] + [
            c for c in ["N_uts", "Pa", "abs_humidity"] if c in crnp_df.columns
        ]
        merged = pd.merge(
            theta_combined,
            crnp_df[keep_cols],
            on="date", how="outer",
        ).sort_values("date").reset_index(drop=True)

        return merged