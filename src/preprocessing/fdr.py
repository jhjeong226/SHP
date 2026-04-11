# src/preprocessing/fdr.py
"""
FDRProcessor
============
METER TEROS 11 센서 데이터를 읽어 FDR_hourly.xlsx 로 출력합니다.

폴더 구조:
  {raw_fdr}/
  ├── HC-N1(z6-05589)(z6-05589)-XXXXXXX/   ← 센서별 하위 폴더
  │   ├── ...-Configuration_1-....csv
  │   └── ...-Configuration_2-....csv
  ├── HC-N2(z6-19848)(...)-XXXXXXX/
  ...

폴더명 파싱:
  "HC-N1(z6-05589)..." → site_id="N1", logger_id="z6-05589"

YAML sensors 섹션:
  센서 위치 정보(lat, lon, distance, bulk_density) 제공
  logger_id 또는 site_id 로 매칭

파일 구조 (3줄 헤더):
  Row 1: 로거 ID, 포트명
  Row 2: 레코드 수, 센서 타입
  Row 3: 컬럼명 (Timestamps, m³/m³ Water Content, ...)
  Row 4+: 데이터  (타임스탬프: "08/23/2024 07:00:00 PM")

출력 컬럼:
  timestamp, site_id, depth_cm, theta_v,
  lat, lon, distance_m, bulk_density
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── 폴더명 파싱 ──────────────────────────────────────────────────────────────

_FOLDER_PATTERN = re.compile(
    r"^[A-Z]+-([A-Z0-9]+)"   # HC-N1  → group(1) = "N1"
    r".*?(z6-\d+)",           # (z6-05589) → group(2) = "z6-05589"
    re.IGNORECASE,
)


def _parse_folder(folder_name: str) -> Tuple[Optional[str], Optional[str]]:
    """폴더명에서 (site_id, logger_id) 추출."""
    m = _FOLDER_PATTERN.match(folder_name)
    if m:
        return m.group(1).upper(), m.group(2).lower()
    return None, None


# ── 단일 CSV 읽기 ─────────────────────────────────────────────────────────────

def _read_single_csv(path: Path) -> pd.DataFrame:
    """3줄 헤더 스킵 후 col 0(timestamp), col 1,3,5(theta_v) 추출."""
    df = pd.read_csv(
        path,
        skiprows=3,
        header=None,
        na_values=["", "NAN", "NaN"],
        encoding="utf-8-sig",
        encoding_errors="replace",
        low_memory=False,
    )

    if df.shape[1] < 6:
        raise ValueError(f"컬럼 수 부족 ({df.shape[1]}개)")

    out = pd.DataFrame({
        "timestamp":  df.iloc[:, 0],
        "theta_v_d1": pd.to_numeric(df.iloc[:, 1], errors="coerce"),
        "theta_v_d2": pd.to_numeric(df.iloc[:, 3], errors="coerce"),
        "theta_v_d3": pd.to_numeric(df.iloc[:, 5], errors="coerce"),
    })

    out["timestamp"] = pd.to_datetime(
        out["timestamp"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce",
    )
    return out.dropna(subset=["timestamp"])


# ── 폴더 내 CSV 병합 ─────────────────────────────────────────────────────────

def _merge_folder(folder: Path) -> pd.DataFrame:
    """폴더 내 모든 CSV를 읽어 병합 + 중복 제거 + 시간 정렬."""
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        return pd.DataFrame()

    frames = []
    for p in csv_files:
        try:
            frames.append(_read_single_csv(p))
            print(f"      ✅ {p.name}: {len(frames[-1]):,}행")
        except Exception as e:
            print(f"      ⚠️  {p.name}: {e}")

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    merged = (merged
              .drop_duplicates(subset=["timestamp"])
              .sort_values("timestamp")
              .reset_index(drop=True))
    removed = before - len(merged)
    if removed:
        print(f"      🔧 중복 제거: {removed:,}행")
    return merged


# ── wide → long 변환 ──────────────────────────────────────────────────────────

def _to_long(df: pd.DataFrame, site_id: str,
             sensor_info: Dict, depths: List[int]) -> pd.DataFrame:
    depth_map = {
        "theta_v_d1": depths[0],
        "theta_v_d2": depths[1],
        "theta_v_d3": depths[2],
    }
    long = df.melt(
        id_vars=["timestamp"],
        value_vars=list(depth_map.keys()),
        var_name="depth_key",
        value_name="theta_v",
    )
    long["depth_cm"]     = long["depth_key"].map(depth_map)
    long["site_id"]      = site_id
    long["lat"]          = sensor_info.get("lat")
    long["lon"]          = sensor_info.get("lon")
    long["distance_m"]   = sensor_info.get("distance", 0)
    long["bulk_density"] = sensor_info.get("bulk_density")

    invalid = (long["theta_v"] < 0) | (long["theta_v"] > 1)
    if invalid.sum():
        print(f"      🔧 물리 범위 초과 → NaN: {invalid.sum():,}개")
        long.loc[invalid, "theta_v"] = np.nan

    return long[["timestamp", "site_id", "depth_cm", "theta_v",
                 "lat", "lon", "distance_m", "bulk_density"]]


# ════════════════════════════════════════════════════════════════════════════

class FDRProcessor:
    """
    fdr 폴더의 센서별 하위 폴더를 탐색해 FDR_hourly.xlsx 저장.

    Parameters
    ----------
    station_id  : str
    station_cfg : dict  (YAML station 섹션)
    input_dir   : 센서별 하위 폴더들이 있는 상위 폴더
    output_dir  : 결과 저장 폴더
    """

    def __init__(self, station_id: str, station_cfg: Dict,
                 input_dir: Path, output_dir: Path):
        self.station_id  = station_id
        self.station_cfg = station_cfg
        self.input_dir   = Path(input_dir)
        self.output_dir  = Path(output_dir)
        self.depths      = station_cfg["sensor_configuration"]["depths"]

    def process(self) -> Path:
        # 1. 하위 폴더 탐색
        sensor_folders = sorted([
            d for d in self.input_dir.iterdir() if d.is_dir()
        ])
        if not sensor_folders:
            raise FileNotFoundError(
                f"센서 하위 폴더 없음: {self.input_dir}"
            )

        print(f"[FDRProcessor] {self.station_id}: "
              f"{len(sensor_folders)}개 센서 폴더 발견")

        # 2. YAML sensors → 위치 정보 맵 구성
        location_map = self._build_location_map()

        # 3. 폴더별 처리
        all_frames: List[pd.DataFrame] = []
        unmatched: List[str] = []

        for folder in sensor_folders:
            site_id, logger_id = _parse_folder(folder.name)

            if site_id is None:
                print(f"\n  ⚠️  파싱 실패: {folder.name}")
                unmatched.append(folder.name)
                continue

            # YAML에서 위치 정보 검색 (site_id 또는 logger_id 로 매칭)
            sensor_info = (location_map.get(site_id)
                           or location_map.get(logger_id)
                           or {})

            if not sensor_info:
                print(f"\n  ℹ️  [{site_id}] YAML 위치 정보 없음 "
                      f"(lat/lon/distance 는 None 으로 저장)")

            print(f"\n  [{site_id}]  logger={logger_id}  "
                  f"folder={folder.name}")

            merged = _merge_folder(folder)
            if merged.empty:
                print(f"    ❌ 유효 데이터 없음, 건너뜀")
                continue

            # bulk_density 기본값: station soil_properties
            if "bulk_density" not in sensor_info:
                sensor_info["bulk_density"] = (
                    self.station_cfg["soil_properties"]["bulk_density"]
                )

            long = _to_long(merged, site_id, sensor_info, self.depths)
            all_frames.append(long)

            print(f"    → {len(merged):,}행 → long {len(long):,}행  "
                  f"({merged['timestamp'].iloc[0]} ~ "
                  f"{merged['timestamp'].iloc[-1]})")

        if not all_frames:
            raise ValueError("처리된 데이터 없음.")

        # 4. 전체 합치기 + 정렬
        result = (pd.concat(all_frames, ignore_index=True)
                    .sort_values(["timestamp", "site_id", "depth_cm"])
                    .reset_index(drop=True))

        # 5. 저장
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"{self.station_id}_FDR_hourly.csv"
        result.to_csv(out_path, index=False, encoding="utf-8-sig")

        self._save_daily_avg(result)
        self._print_summary(result, out_path)
        return out_path

    def _build_location_map(self) -> Dict[str, Dict]:
        """
        YAML sensors → {site_id: info, logger_id: info} 딕셔너리.
        site_id 와 logger_id 둘 다 키로 등록해 유연하게 매칭.
        """
        sensors = self.station_cfg.get("sensors", {})
        loc_map: Dict[str, Dict] = {}
        for site_id, info in sensors.items():
            entry = {
                "lat":          info.get("lat"),
                "lon":          info.get("lon"),
                "distance":     info.get("distance", 0),
                "bulk_density": info.get("bulk_density"),
            }
            loc_map[site_id.upper()] = entry
            lid = info.get("logger_id", "").lower()
            if lid:
                loc_map[lid] = entry
        return loc_map

    def _print_summary(self, df: pd.DataFrame, path: Path) -> None:
        sites  = sorted(df["site_id"].unique())
        depths = sorted(df["depth_cm"].unique())

        print(f"\n[FDRProcessor] 완료 → {path.name}")
        print(f"  기간   : {df['timestamp'].min()} ~ {df['timestamp'].max()}")
        print(f"  총 행수 : {len(df):,}")
        print(f"  사이트 : {sites}")
        print(f"  깊이   : {depths} cm")
        print()

        for site in sites:
            for depth in depths:
                sub = df[(df["site_id"] == site) & (df["depth_cm"] == depth)]
                valid = sub["theta_v"].dropna()
                n_nan = sub["theta_v"].isna().sum()
                if len(valid):
                    print(f"  [{site:>3} {depth:>2}cm]  "
                          f"θ {valid.min():.3f}~{valid.max():.3f}  "
                          f"mean={valid.mean():.3f}  결측={n_nan}")

    def _save_daily_avg(self, df: pd.DataFrame) -> Path:
        """
        일별 깊이별 평균 (wide format) 저장.
        출력: Date | 10cm | 20cm | 30cm
        UTS / Desilet 교정 평가 기준 데이터.
        """
        df = df.copy()
        df["Date"] = pd.to_datetime(df["timestamp"]).dt.date

        # 모든 센서 평균 → 날짜 × 깊이
        daily = (df.groupby(["Date", "depth_cm"])["theta_v"]
                   .mean()
                   .unstack("depth_cm"))

        # 컬럼명: 10 → 10cm 등
        daily.columns = [f"{int(c)}cm" for c in daily.columns]
        daily = daily.reset_index()

        avg_path = self.output_dir / f"{self.station_id}_FDR_daily_avg.xlsx"
        daily.to_excel(avg_path, index=False, engine="openpyxl")
        print(f"  📋 FDR_daily_avg 저장 → {avg_path.name}  "
              f"({len(daily)}일, 컬럼: {list(daily.columns)})")
        return avg_path