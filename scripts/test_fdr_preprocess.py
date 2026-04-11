# scripts/test_fdr_preprocess.py
"""
FDR 전처리 테스트 스크립트
실행: python scripts/test_fdr_preprocess.py --station HC
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.fdr import FDRProcessor
from src.utils.io import get_station_paths, load_config


def main():
    parser = argparse.ArgumentParser(description="FDR 전처리 테스트")
    parser.add_argument("--station", "-s", required=True)
    args = parser.parse_args()

    station_id = args.station
    cfg        = load_config(station_id)
    paths      = get_station_paths(station_id)

    raw_fdr   = paths["raw_fdr"]
    processed = paths["processed"]

    print(f"\n{'='*55}")
    print(f"  FDR 전처리 테스트  —  {station_id}")
    print(f"{'='*55}")
    print(f"  입력 폴더 : {raw_fdr}")
    print(f"  출력 폴더 : {processed}")

    # 하위 폴더(센서별) 확인
    sensor_folders = sorted([d for d in raw_fdr.iterdir() if d.is_dir()])
    if not sensor_folders:
        print(f"\n❌ 센서 하위 폴더 없음: {raw_fdr}")
        sys.exit(1)

    print(f"\n  발견된 센서 폴더 ({len(sensor_folders)}개):")
    for folder in sensor_folders:
        csv_count = len(list(folder.glob("*.csv")))
        print(f"    - {folder.name}  ({csv_count}개 CSV)")

    # ── 전처리 실행 ────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    proc = FDRProcessor(
        station_id  = station_id,
        station_cfg = cfg["station"],
        input_dir   = raw_fdr,
        output_dir  = processed,
    )
    out_path = proc.process()

    # ── 결과 확인 ──────────────────────────────────────────────────────────
    import pandas as pd

    df = pd.read_csv(out_path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(f"\n{'─'*55}")
    print("  결과 미리보기 (처음 6행)")
    print(f"{'─'*55}")
    print(df.head(6).to_string(index=False))

    print(f"\n{'─'*55}")
    print("  사이트 × 깊이별 theta_v 요약")
    print(f"{'─'*55}")
    for site in sorted(df["site_id"].unique()):
        for depth in sorted(df["depth_cm"].unique()):
            sub   = df[(df["site_id"] == site) & (df["depth_cm"] == depth)]
            valid = sub["theta_v"].dropna()
            n_nan = sub["theta_v"].isna().sum()
            pct   = n_nan / len(sub) * 100
            mark  = " ⚠️" if pct > 5 else ""
            if len(valid):
                print(f"  [{site:>3} {depth:>2}cm]  "
                      f"{valid.min():.3f}~{valid.max():.3f}  "
                      f"mean={valid.mean():.3f}  "
                      f"결측={n_nan}({pct:.1f}%){mark}")

    print(f"\n{'─'*55}")
    print("  월별 데이터 수")
    print(f"{'─'*55}")
    n_combo  = df.groupby(["site_id", "depth_cm"]).ngroups
    monthly  = df.groupby(df["timestamp"].dt.to_period("M")).size()
    for period, count in monthly.items():
        expected = pd.Period(period).days_in_month * 24 * n_combo
        pct      = count / expected * 100
        bar      = "█" * int(pct / 5)
        print(f"  {period}  {count:>6}행  {pct:>5.1f}%  {bar}")

    print(f"\n✅ 저장 완료: {out_path}\n")


if __name__ == "__main__":
    main()