# src/calibration/matching.py
"""
DataMatcher
===========
보정된 시간별 CRNP 데이터와 시간별 FDR 데이터를 일별로 매칭합니다.

FDR 대표값:
  crnpy.nrad_weight() 로 거리·깊이 가중평균 (Schron_2017)
  실패 시 단순 평균 fallback

CRNP 대표값:
  일평균 N_corrected

출력 columns: date | theta | N
  → StandardCalibrator / SHPCalibrator 의 matched_df 입력
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class DataMatcher:
    """
    Parameters
    ----------
    station_cfg : YAML station 섹션
    """

    def __init__(self, station_cfg: Dict):
        self.station_cfg  = station_cfg
        self.bulk_density = station_cfg["soil_properties"]["bulk_density"]
        cal_cfg           = station_cfg.get("calibration", {})
        self.weighting    = cal_cfg.get("weighting_method", "Schron_2017")
        self.ref_depths   = cal_cfg.get("reference_depths",
                            station_cfg["sensor_configuration"]["depths"])

    # ── 공개 인터페이스 ──────────────────────────────────────────────────────

    def match(self,
              fdr_df:   pd.DataFrame,
              crnp_df:  pd.DataFrame,
              start:    str,
              end:      str) -> pd.DataFrame:
        """
        지정 기간의 일별 (theta, N) 쌍을 반환합니다.
        start / end 는 FDR-CRNP 겹치는 전체 기간으로 넓게 설정하세요.
        교정 날짜 탐색과 평가는 Calibrator 내부에서 수행합니다.

        Returns
        -------
        DataFrame: date | theta | N | n_sensors
        """
        fdr  = fdr_df.copy()
        crnp = crnp_df.copy()
        fdr["timestamp"]  = pd.to_datetime(fdr["timestamp"])
        crnp["timestamp"] = pd.to_datetime(crnp["timestamp"])

        s, e = pd.to_datetime(start), pd.to_datetime(end)
        fdr  = fdr[ (fdr["timestamp"]  >= s) & (fdr["timestamp"]  <= e)]
        crnp = crnp[(crnp["timestamp"] >= s) & (crnp["timestamp"] <= e)]

        # CRNP 일평균
        crnp["date"] = crnp["timestamp"].dt.date
        daily_crnp = crnp.groupby("date").agg(
            N            = ("N_corrected",  "mean"),
            abs_humidity = ("abs_humidity", "mean"),
            Pa           = ("Pa",           "mean"),
        ).reset_index()

        # FDR 날짜 컬럼
        fdr["date"] = fdr["timestamp"].dt.date

        results: List[Dict] = []
        for _, row in daily_crnp.iterrows():
            d    = row["date"]
            fday = fdr[
                (fdr["date"] == d) &
                (fdr["depth_cm"].isin(self.ref_depths)) &
                fdr["theta_v"].notna() &
                (fdr["theta_v"] > 0) &
                (fdr["theta_v"] < 1)
            ]
            if fday.empty:
                continue

            # 센서×깊이 일평균 → nrad_weight 입력 (36포인트)
            fday_daily = (fday
                          .groupby(["site_id", "depth_cm",
                                    "lat", "lon", "distance_m", "bulk_density"])
                          ["theta_v"].mean()
                          .reset_index())

            theta, n_sensors = self._weighted_mean(fday_daily, row)
            results.append({
                "date":      d,
                "theta":     theta,
                "N":         row["N"],
                "n_sensors": n_sensors,
            })

        matched = pd.DataFrame(results)
        if matched.empty:
            print("  ⚠️  매칭 결과 없음. 기간·파일을 확인하세요.")
            return matched

        print(f"  매칭: {len(matched)}일  "
              f"θ {matched['theta'].min():.3f}~{matched['theta'].max():.3f}  "
              f"N {matched['N'].min():.1f}~{matched['N'].max():.1f}")
        return matched

    # ── 내부 헬퍼 ───────────────────────────────────────────────────────────

    def _weighted_mean(self, fday: pd.DataFrame,
                       crnp_row: pd.Series) -> Tuple[float, int]:
        """crnpy.nrad_weight 가중평균, 실패시 단순평균."""
        try:
            import crnpy
            fday = fday.copy()
            fday["profile_id"] = (fday["lat"].astype(str) + "_" +
                                  fday["lon"].astype(str))
            theta_w, _ = crnpy.nrad_weight(
                abs_humidity = float(crnp_row["abs_humidity"]),
                theta_v      = fday["theta_v"].values,
                distances    = fday["distance_m"].values,
                depths       = fday["depth_cm"].values,
                profiles     = fday["profile_id"].values,
                rhob         = self.bulk_density,
                p            = float(crnp_row["Pa"]),
                method       = self.weighting,
            )
            return float(theta_w), len(fday)
        except Exception:
            return float(fday["theta_v"].mean()), len(fday)