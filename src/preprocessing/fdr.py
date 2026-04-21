# src/preprocessing/fdr.py
"""
FDRProcessor  ──  FDR 전처리 파이프라인
=========================================

모든 파라미터는 config/processing_options.yaml의
preprocessing.fdr 섹션에서 읽어옵니다.

출력 파일 2개:

[1] {station_id}_FDR_hourly.xlsx
    시트: 10cm / 20cm / 30cm
    컬럼: timestamp, {E1}, {E2}, {N1}, ..., mean

[2] {station_id}_FDR_daily.xlsx
    시트: 10cm / 20cm / 30cm
    컬럼: date, {E1}, {E2}, {N1}, ..., mean

폴더 구조 (입력) — 두 가지 구조 자동 감지:

  [subfolder 방식 — HC]
  {raw_fdr}/
  ├── HC-E1(z6-19850)(...)-XXXXXXX/   ← 센서별 하위 폴더
  │   ├── ...-Configuration_1-....csv
  │   └── ...-Configuration_2-....csv

  [flat 방식 — PC]
  {raw_fdr}/
  ├── z6-25663(S25)(z6-25663)-Configuration 2-XXXXXXX.csv  ← 직접 CSV
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# 기본값 (YAML에 해당 키가 없을 때 사용)
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULTS: Dict[str, Any] = {
    "csv_header_rows":    3,
    "csv_theta_cols":     [1, 3, 5],
    "timestamp_format":   "%m/%d/%Y %I:%M:%S %p",
    "filter_on_hour":     True,
    "theta_min":          0.0,
    "theta_max":          1.0,
    "min_obs_daily":      6,
}


def _cfg(options: Dict, key: str) -> Any:
    """options['preprocessing']['fdr'][key] 를 안전하게 읽되, 없으면 기본값 반환."""
    return (options
            .get("preprocessing", {})
            .get("fdr", {})
            .get(key, _DEFAULTS[key]))


# ═══════════════════════════════════════════════════════════════════════════════
# 내부 헬퍼 함수
# ═══════════════════════════════════════════════════════════════════════════════

# HC 방식: HC-E1(z6-19850)... 형태의 폴더/파일명
_FOLDER_PATTERN_HC = re.compile(
    r"^[A-Z]+-([A-Z0-9]+)"
    r".*?(z6-\d+)",
    re.IGNORECASE,
)

# PC 방식: z6-25663(S25)(z6-25663)... 형태의 폴더/파일명
_FOLDER_PATTERN_PC = re.compile(
    r"^(z6-\d+)\(([A-Z0-9]+)\)",
    re.IGNORECASE,
)


def _parse_folder(folder_name: str) -> Tuple[Optional[str], Optional[str]]:
    """폴더명에서 (site_id, logger_id) 추출. HC/PC 두 패턴 모두 지원."""
    m = _FOLDER_PATTERN_HC.match(folder_name)
    if m:
        return m.group(1).upper(), m.group(2).lower()
    m = _FOLDER_PATTERN_PC.match(folder_name)
    if m:
        return m.group(2).upper(), m.group(1).lower()
    return None, None


def _parse_flat_filename(file_name: str) -> Tuple[Optional[str], Optional[str]]:
    """flat 파일명에서 (site_id, logger_id) 추출."""
    m = _FOLDER_PATTERN_PC.match(file_name)
    if m:
        return m.group(2).upper(), m.group(1).lower()
    return None, None


def _read_single_csv(path: Path, options: Dict) -> pd.DataFrame:
    """헤더 스킵 후 timestamp + 3 depth 컬럼 추출."""
    header_rows = _cfg(options, "csv_header_rows")
    c1, c2, c3  = _cfg(options, "csv_theta_cols")
    ts_fmt       = _cfg(options, "timestamp_format")
    on_hour      = _cfg(options, "filter_on_hour")
    min_col      = max(c1, c2, c3) + 1

    df = pd.read_csv(
        path,
        skiprows=header_rows,
        header=None,
        na_values=["", "NAN", "NaN", "nan"],
        encoding="utf-8-sig",
        encoding_errors="replace",
        low_memory=False,
    )

    if df.shape[1] < min_col:
        raise ValueError(f"컬럼 수 부족: {df.shape[1]}개 (최소 {min_col}개 필요)")

    out = pd.DataFrame({
        "timestamp":  df.iloc[:, 0],
        "theta_v_d1": pd.to_numeric(df.iloc[:, c1], errors="coerce"),
        "theta_v_d2": pd.to_numeric(df.iloc[:, c2], errors="coerce"),
        "theta_v_d3": pd.to_numeric(df.iloc[:, c3], errors="coerce"),
    })
    out["timestamp"] = pd.to_datetime(out["timestamp"], format=ts_fmt, errors="coerce")
    out = out.dropna(subset=["timestamp"])

    if on_hour:
        out = out[out["timestamp"].dt.minute == 0].copy()
    return out


def _merge_folder(folder: Path, options: Dict) -> pd.DataFrame:
    """폴더 내 -Configuration- 포함 CSV만 병합 + 중복 제거 + 시간 정렬.
    -Metadata- 등 데이터가 아닌 파일은 자동 제외."""
    all_csv = sorted(folder.glob("*.csv"))
    # -Configuration 포함 파일만 읽음 (Metadata, 설정파일 등 제외)
    csv_files = [f for f in all_csv if "-Configuration" in f.name]
    if not csv_files:
        # fallback: Configuration 파일이 없으면 전체 시도
        csv_files = all_csv
    if not csv_files:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for p in csv_files:
        try:
            frames.append(_read_single_csv(p, options))
        except Exception as e:
            print(f"      ⚠️  {p.name} 읽기 실패: {e}")

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


def _pivot_and_write(writer: pd.ExcelWriter,
                     sub: pd.DataFrame,
                     time_col: str,
                     site_ids: List[str],
                     depth: int) -> None:
    """Long DataFrame → Wide pivot → Excel 시트 기록."""
    pivot = (
        sub.pivot_table(index=time_col, columns="site_id",
                        values="theta_v", aggfunc="mean")
        .reindex(columns=site_ids)
    )
    pivot.columns.name = None
    pivot["mean"] = pivot[site_ids].mean(axis=1)
    pivot = pivot.reset_index()
    pivot[time_col] = pivot[time_col].astype(str)
    pivot.to_excel(writer, sheet_name=f"{depth}cm", index=False)

    n_valid = pivot[site_ids].notna().sum().sum()
    print(f"  ✅ Sheet {depth}cm  "
          f"({len(pivot):,}행 × {len(site_ids)}지점 / 유효값 {n_valid:,}개)")


# ═══════════════════════════════════════════════════════════════════════════════
# FDRProcessor
# ═══════════════════════════════════════════════════════════════════════════════

class FDRProcessor:
    """
    FDR 전처리 → hourly / daily Excel 2개 생성.

    Parameters
    ----------
    station_id  : str
    station_cfg : dict   (YAML station 섹션)
    options     : dict   (YAML processing_options 섹션)
    input_dir   : Path   (센서별 하위 폴더들의 상위 폴더)
    output_dir  : Path   (결과 저장 루트)
    """

    def __init__(self, station_id: str, station_cfg: Dict, options: Dict,
                 input_dir: Path, output_dir: Path):
        self.station_id  = station_id
        self.station_cfg = station_cfg
        self.options     = options
        self.input_dir   = Path(input_dir)
        self.output_dir  = Path(output_dir)
        self.depths: List[int] = station_cfg["sensor_configuration"]["depths"]
        self.fdr_dir = self.output_dir / "fdr"

    # ─────────────────────────────────────────────────────────────────────────
    # 공개 인터페이스
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Path]:
        """
        전체 실행.

        Returns
        -------
        dict:  "hourly" → Path,  "daily" → Path
        """
        print(f"\n{'='*60}")
        print(f"  FDRProcessor  ─  {self.station_id}")
        self._print_settings()
        print(f"{'='*60}")

        all_long = self._collect_all_sites()
        site_ids = sorted(all_long["site_id"].unique())

        print(f"\n  활성 지점: {site_ids}  ({len(site_ids)}개)")
        print(f"  기간: {all_long['timestamp'].min().date()} "
              f"~ {all_long['timestamp'].max().date()}")

        print("\n[Step 1] Hourly 파일 생성")
        hourly_path = self._build_hourly(all_long, site_ids)

        print("\n[Step 2] Daily 파일 생성")
        daily_path = self._build_daily(all_long, site_ids)

        print(f"\n{'='*60}")
        print(f"  ✅ FDR 전처리 완료  ─  {self.station_id}")
        print(f"{'='*60}\n")

        return {"hourly": hourly_path, "daily": daily_path}

    # ─────────────────────────────────────────────────────────────────────────
    # 원시 데이터 수집
    # ─────────────────────────────────────────────────────────────────────────

    def _collect_all_sites(self) -> pd.DataFrame:
        """
        입력 폴더 구조를 자동 감지해서 Long DataFrame으로 반환.
          - CSV 파일이 직접 있으면 → flat 방식 (PC)
          - 하위 폴더가 있으면    → subfolder 방식 (HC)
        """
        has_csv     = any(self.input_dir.glob("*.csv"))
        has_folders = any(d.is_dir() for d in self.input_dir.iterdir()
                          if not d.name.startswith("."))

        if has_csv and not has_folders:
            print(f"\n  📂 flat 방식 감지 (CSV 직접 위치)")
            return self._collect_flat()
        elif has_folders:
            print(f"\n  📂 subfolder 방식 감지")
            return self._collect_subfolders()
        else:
            raise FileNotFoundError(
                f"CSV 파일 또는 하위 폴더 없음: {self.input_dir}"
            )

    def _collect_subfolders(self) -> pd.DataFrame:
        """HC 방식: 센서별 하위 폴더 순회."""
        sensor_folders = sorted([d for d in self.input_dir.iterdir() if d.is_dir()])
        print(f"  발견된 센서 폴더: {len(sensor_folders)}개")

        sources = []
        for folder in sensor_folders:
            parsed = _parse_folder(folder.name)
            sources.append((folder.name, parsed,
                             lambda f=folder: _merge_folder(f, self.options)))
        return self._process_sources(sources)

    def _collect_flat(self) -> pd.DataFrame:
        """PC 방식: fdr 폴더 내 CSV 파일 직접 순회. 같은 site_id 파일을 묶어 처리."""
        csv_files = sorted(self.input_dir.glob("*.csv"))
        print(f"  발견된 CSV 파일: {len(csv_files)}개")

        # site_id 기준으로 파일 묶기
        sensor_map: Dict[str, Dict] = {}
        for f in csv_files:
            site_id, logger_id = _parse_flat_filename(f.name)
            if site_id is None:
                print(f"  ⚠️  파일명 파싱 실패 (건너뜀): {f.name}")
                continue
            if site_id not in sensor_map:
                sensor_map[site_id] = {"site_id": site_id,
                                        "logger_id": logger_id, "files": []}
            sensor_map[site_id]["files"].append(f)

        sources = []
        for info in sensor_map.values():
            files = info["files"]
            sources.append((
                info["site_id"],
                (info["site_id"], info["logger_id"]),
                lambda fs=files: self._merge_flat_files(fs),
            ))
        return self._process_sources(sources)

    def _merge_flat_files(self, files: List[Path]) -> pd.DataFrame:
        """flat 방식: 같은 센서의 여러 CSV 파일을 읽어 병합."""
        frames = []
        for p in sorted(files):
            try:
                frames.append(_read_single_csv(p, self.options))
            except Exception as e:
                print(f"      ⚠️  {p.name} 읽기 실패: {e}")
        if not frames:
            return pd.DataFrame()
        merged = pd.concat(frames, ignore_index=True)
        before = len(merged)
        merged = (merged.drop_duplicates(subset=["timestamp"])
                  .sort_values("timestamp").reset_index(drop=True))
        removed = before - len(merged)
        if removed:
            print(f"      🔧 중복 제거: {removed:,}행")
        return merged

    def _process_sources(self, sources) -> pd.DataFrame:
        """(name, (site_id, logger_id), data_fn) 목록을 처리해 Long DataFrame 반환."""
        theta_min  = _cfg(self.options, "theta_min")
        theta_max  = _cfg(self.options, "theta_max")
        depth_keys = ["theta_v_d1", "theta_v_d2", "theta_v_d3"]
        depth_map  = dict(zip(depth_keys, self.depths))
        frames: List[pd.DataFrame] = []

        for name, (site_id, logger_id), data_fn in sources:
            if site_id is None:
                print(f"  ⚠️  파싱 실패 (건너뜀): {name}")
                continue

            print(f"\n  [{site_id}]  logger={logger_id}")
            wide = data_fn()
            if wide.empty:
                print(f"    ❌ 유효 데이터 없음, 건너뜀")
                continue

            long = wide.melt(id_vars=["timestamp"], value_vars=depth_keys,
                             var_name="depth_key", value_name="theta_v")
            long["depth_cm"] = long["depth_key"].map(depth_map)
            long["site_id"]  = site_id
            long = long.drop(columns="depth_key")

            invalid = (long["theta_v"] < theta_min) | (long["theta_v"] > theta_max)
            if invalid.sum():
                print(f"    🔧 물리 범위 이탈 → NaN: {invalid.sum():,}개")
                long.loc[invalid, "theta_v"] = np.nan

            frames.append(long[["timestamp", "site_id", "depth_cm", "theta_v"]])
            n = len(wide)
            print(f"    ✅ {n:,}행  "
                  f"({wide['timestamp'].iloc[0].date()} ~ "
                  f"{wide['timestamp'].iloc[-1].date()})")

        if not frames:
            raise ValueError("처리된 데이터 없음. 입력 폴더를 확인하세요.")

        return (pd.concat(frames, ignore_index=True)
                .sort_values(["timestamp", "site_id", "depth_cm"])
                .reset_index(drop=True))

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Hourly Excel
    # ─────────────────────────────────────────────────────────────────────────

    def _build_hourly(self, df: pd.DataFrame, site_ids: List[str]) -> Path:
        self.fdr_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.fdr_dir / f"{self.station_id}_FDR_hourly.xlsx"

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            for depth in self.depths:
                sub = df[df["depth_cm"] == depth]
                _pivot_and_write(writer, sub, "timestamp", site_ids, depth)

        print(f"  📄 {out_path.name}")
        return out_path

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Daily Excel
    # ─────────────────────────────────────────────────────────────────────────

    def _build_daily(self, df: pd.DataFrame, site_ids: List[str]) -> Path:
        min_obs = _cfg(self.options, "min_obs_daily")

        self.fdr_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.fdr_dir / f"{self.station_id}_FDR_daily.xlsx"

        df_copy = df.copy()
        df_copy["date"] = df_copy["timestamp"].dt.date

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            for depth in self.depths:
                sub = df_copy[df_copy["depth_cm"] == depth]
                grp = sub.groupby(["date", "site_id"])["theta_v"]
                daily_long = (grp.mean()
                              .where(grp.count() >= min_obs)
                              .reset_index())
                _pivot_and_write(writer, daily_long, "date", site_ids, depth)

        print(f"  📄 {out_path.name}")
        return out_path

    # ─────────────────────────────────────────────────────────────────────────
    # 설정값 출력
    # ─────────────────────────────────────────────────────────────────────────

    def _print_settings(self) -> None:
        keys = list(_DEFAULTS.keys())
        print("\n  [설정값]")
        for k in keys:
            print(f"    {k:<22}: {_cfg(self.options, k)}")