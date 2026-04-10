# src/preprocessing/crnp.py
"""
CRNPProcessor
=============
Campbell Scientific TOA5 형식의 .dat 파일을 읽어
hourly_CRNP.xlsx 로 출력합니다.

TOA5 헤더 구조 (4줄):
  Row 1: 기기 정보 (무시)
  Row 2: 컬럼명
  Row 3: 단위   (무시)
  Row 4: 처리방식 (무시)
  Row 5~: 데이터

필수 입력 컬럼:
  TIMESTAMP, Air_Temp_Avg, RH_Avg, Air_Press_Avg, HI_NeutronCts_Tot

출력 컬럼 (hourly_CRNP.xlsx):
  timestamp       : 시간 (datetime)
  N_counts        : 시간당 중성자 계수 (counts/hr)
  Ta              : 기온 (°C)
  RH              : 상대습도 (%)
  Pa              : 기압 (hPa)
  abs_humidity    : 절대습도 (g/m³) — 자동 계산
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import warnings

import numpy as np
import pandas as pd


# ── 절대습도 계산 ────────────────────────────────────────────────────────────

def _calc_abs_humidity(Ta: pd.Series, RH: pd.Series) -> pd.Series:
    """
    절대습도 (g/m³) 계산.
    Magnus 근사식 → 포화수증기압 → 실제 수증기압 → 절대습도
    """
    # 포화수증기압 [hPa]
    es = 6.112 * np.exp(17.502 * Ta / (Ta + 240.97))
    # 실제 수증기압 [hPa]
    e  = es * RH / 100.0
    # 절대습도 [g/m³]  (수증기 기체 상수 Rv = 461.5 J/(kg·K))
    abs_hum = (e * 100.0) / (461.5 * (Ta + 273.15)) * 1000.0
    return abs_hum


# ── 단일 .dat 파일 읽기 ──────────────────────────────────────────────────────

def _read_single_dat(path: Path) -> pd.DataFrame:
    """
    TOA5 .dat 파일 1개를 읽어 정규화된 DataFrame 반환.
    헤더 4줄 건너뛰고, 2번째 줄(컬럼명)만 사용.
    """
    # 컬럼명만 먼저 추출 (2번째 줄, index=1)
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [f.readline() for _ in range(4)]

    col_line = lines[1].strip().replace('"', '')
    columns  = [c.strip() for c in col_line.split(',')]

    # 데이터 읽기 (4줄 헤더 스킵)
    df = pd.read_csv(
        path,
        skiprows=4,
        header=None,
        names=columns,
        na_values=["NAN", "NaN", "", "nan"],
        encoding="utf-8",
        encoding_errors="replace",
    )

    # TIMESTAMP → datetime
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    df = df.dropna(subset=["TIMESTAMP"])

    return df


# ── 필수 컬럼 확인 ───────────────────────────────────────────────────────────

_REQUIRED_COLS = {
    "TIMESTAMP":         "timestamp",
    "HI_NeutronCts_Tot": "N_counts",
    "Air_Temp_Avg":      "Ta",
    "RH_Avg":            "RH",
    "Air_Press_Avg":     "Pa",
}


def _check_and_rename(df: pd.DataFrame, path: Path) -> Optional[pd.DataFrame]:
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        warnings.warn(f"[CRNPProcessor] 필수 컬럼 누락 ({path.name}): {missing}")
        return None
    return df.rename(columns=_REQUIRED_COLS)


# ════════════════════════════════════════════════════════════════════════════

class CRNPProcessor:
    """
    폴더 내 모든 .dat 파일을 읽어 hourly_CRNP.xlsx 로 저장.

    Parameters
    ----------
    station_id : str
    input_dir  : .dat 파일이 있는 폴더
    output_dir : 결과 저장 폴더
    """

    def __init__(self, station_id: str, input_dir: Path, output_dir: Path):
        self.station_id = station_id
        self.input_dir  = Path(input_dir)
        self.output_dir = Path(output_dir)

    # ── 공개 인터페이스 ──────────────────────────────────────────────────────

    def process(self) -> Path:
        """
        전처리 실행.

        Returns
        -------
        Path : 저장된 hourly_CRNP.xlsx 경로
        """
        dat_files = sorted(self.input_dir.glob("*.dat"))
        if not dat_files:
            raise FileNotFoundError(
                f"No .dat files found in: {self.input_dir}"
            )

        print(f"[CRNPProcessor] {self.station_id}: {len(dat_files)}개 .dat 파일 발견")

        # 1. 모든 파일 읽기
        frames: List[pd.DataFrame] = []
        for f in dat_files:
            raw = _read_single_dat(f)
            renamed = _check_and_rename(raw, f)
            if renamed is not None:
                frames.append(renamed)
                print(f"  ✅ {f.name}: {len(renamed):,}행")
            else:
                print(f"  ⚠️  {f.name}: 건너뜀 (필수 컬럼 누락)")

        if not frames:
            raise ValueError("읽기 성공한 .dat 파일이 없습니다.")

        # 2. 병합 + 중복 제거 + 시간 정렬
        df = pd.concat(frames, ignore_index=True)
        before = len(df)
        df = (df
              .drop_duplicates(subset=["timestamp"])
              .sort_values("timestamp")
              .reset_index(drop=True))
        after = len(df)

        if before - after > 0:
            print(f"  🔧 중복 제거: 총 {before:,}행 중 {before - after:,}행 제거 → {after:,}행 남음")

        # 3. 수치 변환 및 물리적 이상값 제거
        df = self._clean(df)

        # 4. 절대습도 계산
        df["abs_humidity"] = _calc_abs_humidity(df["Ta"], df["RH"])

        # 5. 출력 컬럼 정리
        out_cols = ["timestamp", "N_counts", "Ta", "RH", "Pa", "abs_humidity"]
        df_out = df[out_cols].copy()

        # 6. 저장
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"{self.station_id}_CRNP_hourly.xlsx"
        df_out.to_excel(out_path, index=False, engine="openpyxl")

        # 7. 요약 출력
        self._print_summary(df_out, out_path)

        return out_path

    # ── 내부 헬퍼 ───────────────────────────────────────────────────────────

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """수치 변환 + 물리적 이상값 → NaN"""
        for col in ["N_counts", "Ta", "RH", "Pa"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 물리 범위 필터
        df.loc[df["N_counts"] < 0,                   "N_counts"] = np.nan
        df.loc[df["Ta"].abs() > 60,                   "Ta"]       = np.nan
        df.loc[(df["RH"] < 0) | (df["RH"] > 105),    "RH"]       = np.nan
        df.loc[(df["Pa"] < 500) | (df["Pa"] > 1100),  "Pa"]       = np.nan

        return df

    def _print_summary(self, df: pd.DataFrame, path: Path) -> None:
        valid_N = df["N_counts"].notna().sum()
        print(f"\n[CRNPProcessor] 전처리 완료 → {path.name}")
        print(f"  기간  : {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}")
        print(f"  총행수: {len(df):,}  (유효 N_counts: {valid_N:,})")
        print(f"  N_counts: {df['N_counts'].min():.0f} ~ {df['N_counts'].max():.0f}  "
              f"(mean={df['N_counts'].mean():.1f})")
        print(f"  Ta      : {df['Ta'].min():.1f} ~ {df['Ta'].max():.1f} °C")
        print(f"  RH      : {df['RH'].min():.1f} ~ {df['RH'].max():.1f} %")
        print(f"  Pa      : {df['Pa'].min():.1f} ~ {df['Pa'].max():.1f} hPa")
        print(f"  abs_hum : {df['abs_humidity'].min():.2f} ~ "
              f"{df['abs_humidity'].max():.2f} g/m³")