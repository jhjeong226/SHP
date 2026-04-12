# src/preprocessing/crnp.py
"""
CRNPProcessor  ──  CRNP 원시자료 전처리 + 중성자 보정 + 일자료 생성
=====================================================================

모든 파라미터는 config/processing_options.yaml의
corrections 및 preprocessing.crnp 섹션에서 읽어옵니다.

[Step 1] 원시자료 읽기
  Campbell Scientific TOA5 .dat 파일 → 시간 단위 DataFrame

[Step 2] 중성자 보정
  N_corrected = N_raw × fw / (fp × fi)
    fi : 입사 우주선 강도 보정  (NMDB)
    fp : 기압 보정              (Zreda et al., 2012)
    fw : 절대습도 보정           (Rosolem et al., 2013)

[Step 3] 일자료 출력
  → {station_id}_CRNP_daily.xlsx
    컬럼: date, N_raw, N_corrected, Pa, RH, Ta, abs_humidity,
          fi, fp, fw, Pref, Aref, Iref

TOA5 헤더 구조 (4줄):
  Row 1: 기기 정보
  Row 2: 컬럼명
  Row 3: 단위
  Row 4: 처리방식
  Row 5+: 데이터
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# 기본값 (YAML에 해당 키가 없을 때 사용)
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULTS: Dict[str, Any] = {
    "ta_range":               [-40,  60],
    "rh_range":               [  0, 110],
    "pa_range":               [500, 1100],
    "interp_method":          "pchip",
    "interp_limit_hours":     24,
    "pressure_scale_height":  130.0,
    "humidity_coeff":         0.0054,
    "mad_threshold":          3.0,
    "min_obs_daily":          6,
    "dat_file_pattern":       None,
    "n_daily_min":            None,
    "n_daily_max":            None,
}


def _cfg(options: Dict, key: str) -> Any:
    """options['preprocessing']['crnp'][key] 를 안전하게 읽되, 없으면 기본값 반환."""
    return (options
            .get("preprocessing", {})
            .get("crnp", {})
            .get(key, _DEFAULTS[key]))


# ═══════════════════════════════════════════════════════════════════════════════
# 내부 헬퍼 함수
# ═══════════════════════════════════════════════════════════════════════════════

_REQUIRED_COLS: Dict[str, str] = {
    "TIMESTAMP":          "timestamp",
    "HI_NeutronCts_Tot":  "N_raw",
    "Air_Temp_Avg":       "Ta",
    "RH_Avg":             "RH",
    "Air_Press_Avg":      "Pa",
}


def _read_single_dat(path: Path) -> pd.DataFrame:
    """TOA5 .dat 파일 1개 읽기 (4줄 헤더 스킵)."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [f.readline() for _ in range(4)]

    col_line = lines[1].strip().replace('"', '')
    columns  = [c.strip() for c in col_line.split(',')]

    df = pd.read_csv(
        path,
        skiprows=4,
        header=None,
        names=columns,
        na_values=["NAN", "NaN", "", "nan"],
        encoding="utf-8",
        encoding_errors="replace",
        low_memory=False,
    )
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    return df.dropna(subset=["TIMESTAMP"])


def _check_and_rename(df: pd.DataFrame, path: Path) -> Optional[pd.DataFrame]:
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        warnings.warn(f"필수 컬럼 누락 ({path.name}): {missing}")
        return None
    return df.rename(columns=_REQUIRED_COLS)


def _abs_humidity(Ta: pd.Series, RH: pd.Series) -> pd.Series:
    """절대습도 [g/m³] — Magnus 근사식."""
    es = 6.112 * np.exp(17.502 * Ta / (Ta + 240.97))
    e  = es * RH / 100.0
    return (e * 100.0) / (461.5 * (Ta + 273.15)) * 1000.0


def _remove_outliers_mad(series: pd.Series, threshold: float) -> pd.Series:
    """MAD 기반 이상치 → NaN."""
    median = series.median()
    mad    = (series - median).abs().median()
    if mad < 1e-9:
        return series
    score = (series - median).abs() / (mad + 1e-9)
    return series.where(score < threshold)


# ═══════════════════════════════════════════════════════════════════════════════
# CRNPProcessor
# ═══════════════════════════════════════════════════════════════════════════════

class CRNPProcessor:
    """
    CRNP 원시자료 전처리 + 중성자 보정 + 일자료 생성.

    Parameters
    ----------
    station_id  : str
    station_cfg : dict   (YAML station 섹션)
    options     : dict   (YAML processing_options 섹션)
    input_dir   : Path   (.dat 파일 폴더)
    output_dir  : Path   (결과 저장 루트)
    """

    def __init__(self, station_id: str, station_cfg: Dict, options: Dict,
                 input_dir: Path, output_dir: Path):
        self.station_id  = station_id
        self.station_cfg = station_cfg
        self.options     = options
        self.input_dir   = Path(input_dir)
        self.output_dir  = Path(output_dir)

        cal = station_cfg.get("calibration", {})
        self.nmdb_station: str = cal.get("neutron_monitor", "MXCO")
        self.utc_offset:   int = cal.get("utc_offset", 9)

        corr = options.get("corrections", {})
        self.do_fi: bool = corr.get("incoming_flux", True)
        self.do_fp: bool = corr.get("pressure",      True)
        self.do_fw: bool = corr.get("humidity",       True)

        # Step 2에서 산출되는 참조값
        self.Pref: Optional[float] = None
        self.Aref: Optional[float] = None
        self.Iref: Optional[float] = None

    # ─────────────────────────────────────────────────────────────────────────
    # 공개 인터페이스
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> Path:
        """
        전체 파이프라인 실행.

        Returns
        -------
        Path : 저장된 {station_id}_CRNP_daily.xlsx 경로
        """
        print(f"\n{'='*60}")
        print(f"  CRNPProcessor  ─  {self.station_id}")
        self._print_settings()
        print(f"{'='*60}")

        print("\n[Step 1] 원시자료 읽기")
        df = self._step1_read()

        print("\n[Step 2] 중성자 보정")
        df = self._step2_correct(df)

        print("\n[Step 3] 일자료 생성 및 저장")
        out_path = self._step3_daily(df)

        print(f"\n{'='*60}")
        print(f"  ✅ CRNP 전처리 완료  ─  {self.station_id}")
        print(f"  기간  : {df['timestamp'].min().date()} "
              f"~ {df['timestamp'].max().date()}")
        ref_str = (f"Pref={self.Pref:.1f} hPa  "
                   f"Aref={self.Aref:.4f} g/m³  "
                   f"Iref={self.Iref:.1f}" if self.Iref else
                   f"Pref={self.Pref:.1f} hPa  Aref={self.Aref:.4f} g/m³")
        print(f"  {ref_str}")
        print(f"{'='*60}\n")
        return out_path

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: 원시자료 읽기
    # ─────────────────────────────────────────────────────────────────────────

    def _step1_read(self) -> pd.DataFrame:
        pattern = _cfg(self.options, "dat_file_pattern")
        dat_files = sorted(self.input_dir.glob("*.dat"))
        if pattern:
            dat_files = [f for f in dat_files
                         if pattern.lower() in f.name.lower()]
            print(f"  파일 필터 적용: '{pattern}'  → {len(dat_files)}개 매칭")
        if not dat_files:
            raise FileNotFoundError(
                f".dat 파일 없음 (pattern={pattern!r}): {self.input_dir}"
            )

        print(f"  .dat 파일: {len(dat_files)}개")
        frames: List[pd.DataFrame] = []
        for f in dat_files:
            raw     = _read_single_dat(f)
            renamed = _check_and_rename(raw, f)
            if renamed is not None:
                frames.append(renamed)
                print(f"  ✅ {f.name}: {len(renamed):,}행")
            else:
                print(f"  ⚠️  {f.name}: 건너뜀")

        if not frames:
            raise ValueError("읽기 성공한 .dat 파일이 없습니다.")

        df = pd.concat(frames, ignore_index=True)
        before = len(df)
        df = (df.drop_duplicates(subset=["timestamp"])
              .sort_values("timestamp")
              .reset_index(drop=True))
        removed = before - len(df)
        if removed:
            print(f"  🔧 중복 제거: {removed:,}행")

        df = self._clean(df)
        df["abs_humidity"] = _abs_humidity(df["Ta"], df["RH"])

        print(f"  총 {len(df):,}행  "
              f"({df['timestamp'].iloc[0].date()} ~ "
              f"{df['timestamp'].iloc[-1].date()})")
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """수치 변환 + 물리 범위 이상치 → NaN + 기상 변수 보간."""
        for col in ["N_raw", "Ta", "RH", "Pa"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        ta_min, ta_max = _cfg(self.options, "ta_range")
        rh_min, rh_max = _cfg(self.options, "rh_range")
        pa_min, pa_max = _cfg(self.options, "pa_range")

        df.loc[df["N_raw"] < 0,       "N_raw"] = np.nan
        df.loc[df["Ta"]  < ta_min,    "Ta"]    = np.nan
        df.loc[df["Ta"]  > ta_max,    "Ta"]    = np.nan
        df.loc[df["RH"]  < rh_min,    "RH"]    = np.nan
        df.loc[df["RH"]  > rh_max,    "RH"]    = np.nan
        df.loc[df["Pa"]  < pa_min,    "Pa"]    = np.nan
        df.loc[df["Pa"]  > pa_max,    "Pa"]    = np.nan

        method = _cfg(self.options, "interp_method")
        limit  = _cfg(self.options, "interp_limit_hours")
        for col in ["Ta", "RH", "Pa"]:
            if col in df.columns:
                df[col] = df[col].interpolate(
                    method=method, limit=limit, limit_direction="both"
                )
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: 중성자 보정
    # ─────────────────────────────────────────────────────────────────────────

    def _step2_correct(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        self.Pref = float(df["Pa"].mean())
        self.Aref = float(df["abs_humidity"].mean())

        # fi: 입사 우주선 강도 보정
        if self.do_fi:
            df = self._apply_fi(df)
        else:
            df["fi"] = 1.0
            self.Iref = np.nan
            print("  ℹ️  fi 보정 비활성화 (fi=1)")

        # fp: 기압 보정
        if self.do_fp:
            L = _cfg(self.options, "pressure_scale_height")
            # fp = exp((Pref - Pa) / L)
            # 고기압(Pa > Pref) → fp < 1 → N_corrected 상향 (대기 두꺼워 중성자 감소)
            # 저기압(Pa < Pref) → fp > 1 → N_corrected 하향
            df["fp"] = np.exp((self.Pref - df["Pa"]) / L)
            print(f"  ✅ fp 보정 완료  (Pref={self.Pref:.1f} hPa, L={L})")
        else:
            df["fp"] = 1.0
            print("  ℹ️  fp 보정 비활성화 (fp=1)")

        # fw: 절대습도 보정
        if self.do_fw:
            coeff = _cfg(self.options, "humidity_coeff")
            df["fw"] = 1.0 + coeff * (df["abs_humidity"] - self.Aref)
            print(f"  ✅ fw 보정 완료  (Aref={self.Aref:.4f} g/m³, coeff={coeff})")
        else:
            df["fw"] = 1.0
            print("  ℹ️  fw 보정 비활성화 (fw=1)")

        # 보정 중성자 계산
        df["N_corrected"] = df["N_raw"] * df["fw"] / (df["fp"] * df["fi"])

        # MAD 이상치 제거
        threshold = _cfg(self.options, "mad_threshold")
        before    = df["N_corrected"].notna().sum()
        df["N_corrected"] = _remove_outliers_mad(df["N_corrected"], threshold)
        removed = before - df["N_corrected"].notna().sum()
        if removed:
            print(f"  🔧 N_corrected MAD 이상치 제거: {removed}개 (threshold={threshold})")

        valid = df["N_corrected"].dropna()
        print(f"  ✅ N_corrected  "
              f"min={valid.min():.0f}  mean={valid.mean():.0f}  max={valid.max():.0f}")
        return df

    def _apply_fi(self, df: pd.DataFrame) -> pd.DataFrame:
        """NMDB에서 입사 우주선 강도 다운로드 후 fi 계산."""
        t_min = df["timestamp"].min()
        t_max = df["timestamp"].max()
        try:
            import crnpy
            print(f"  📡 NMDB 다운로드  "
                  f"(station={self.nmdb_station}, "
                  f"{t_min.date()} ~ {t_max.date()})")
            nmdb = crnpy.get_incoming_neutron_flux(
                t_min, t_max,
                station=self.nmdb_station,
                utc_offset=self.utc_offset,
            )
            df["incoming_flux"] = crnpy.interpolate_incoming_flux(
                nmdb["timestamp"], nmdb["counts"], df["timestamp"]
            )
            self.Iref = float(df["incoming_flux"].mean())
            df["fi"]  = crnpy.correction_incoming_flux(
                incoming_neutrons=df["incoming_flux"],
                incoming_Ref=self.Iref,
            )
            print(f"  ✅ fi 보정 완료  (Iref={self.Iref:.1f})")
        except Exception as e:
            warnings.warn(f"NMDB 실패 → fi=1 로 대체: {e}", stacklevel=2)
            df["incoming_flux"] = np.nan
            df["fi"]            = 1.0
            self.Iref           = np.nan
            print(f"  ⚠️  NMDB 실패 → fi=1  ({e})")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: 일자료 저장
    # ─────────────────────────────────────────────────────────────────────────

    def _step3_daily(self, df: pd.DataFrame) -> Path:
        min_obs   = _cfg(self.options, "min_obs_daily")
        n_day_min = _cfg(self.options, "n_daily_min")
        n_day_max = _cfg(self.options, "n_daily_max")

        df_copy = df.copy()
        df_copy["date"] = df_copy["timestamp"].dt.date

        agg_cols = [c for c in
                    ["N_raw", "N_corrected", "Pa", "RH", "Ta",
                     "abs_humidity", "fi", "fp", "fw"]
                    if c in df_copy.columns]

        daily = df_copy.groupby("date")[agg_cols].mean()

        for col in ["N_raw", "N_corrected"]:
            if col in df_copy.columns:
                cnt = df_copy.groupby("date")[col].count()
                daily[col] = daily[col].where(cnt >= min_obs)

        # 일자료 N_corrected 절대값 필터 (YAML n_daily_min / n_daily_max)
        if n_day_min is not None and "N_corrected" in daily.columns:
            below = (daily["N_corrected"] < n_day_min).sum()
            daily.loc[daily["N_corrected"] < n_day_min, "N_corrected"] = np.nan
            if below:
                print(f"  🔧 N_corrected 하한 필터 ({n_day_min}): {below}일 제거")

        if n_day_max is not None and "N_corrected" in daily.columns:
            above = (daily["N_corrected"] > n_day_max).sum()
            daily.loc[daily["N_corrected"] > n_day_max, "N_corrected"] = np.nan
            if above:
                print(f"  🔧 N_corrected 상한 필터 ({n_day_max}): {above}일 제거")

        daily = daily.reset_index()
        daily["date"] = daily["date"].astype(str)

        # 참조값 컬럼 추가
        daily["Pref"] = round(self.Pref, 3) if self.Pref is not None else np.nan
        daily["Aref"] = round(self.Aref, 6) if self.Aref is not None else np.nan
        daily["Iref"] = round(self.Iref, 3) if self.Iref is not None else np.nan

        col_order = ["date", "N_raw", "N_corrected",
                     "Pa", "RH", "Ta", "abs_humidity",
                     "fi", "fp", "fw",
                     "Pref", "Aref", "Iref"]
        col_order = [c for c in col_order if c in daily.columns]
        daily = daily[col_order]

        crnp_dir = self.output_dir / "crnp"
        crnp_dir.mkdir(parents=True, exist_ok=True)
        out_path = crnp_dir / f"{self.station_id}_CRNP_daily.xlsx"
        daily.to_excel(out_path, index=False, engine="openpyxl")

        n_valid = daily["N_corrected"].notna().sum()
        print(f"  ✅ {out_path.name}  ({len(daily)}일 / 유효 {n_valid}일)")
        return out_path

    # ─────────────────────────────────────────────────────────────────────────
    # 설정값 출력
    # ─────────────────────────────────────────────────────────────────────────

    def _print_settings(self) -> None:
        corr = self.options.get("corrections", {})
        print("\n  [보정 on/off]")
        print(f"    fi (incoming flux) : {corr.get('incoming_flux', True)}")
        print(f"    fp (pressure)      : {corr.get('pressure',      True)}")
        print(f"    fw (humidity)      : {corr.get('humidity',       True)}")
        print("\n  [수치 파라미터]")
        for k in _DEFAULTS:
            print(f"    {k:<24}: {_cfg(self.options, k)}")