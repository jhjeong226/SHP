# scripts/test_crnp_preprocess.py
"""
CRNP 전처리 테스트 스크립트
실행: python scripts/test_crnp_preprocess.py --station HC
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.crnp import CRNPProcessor
from src.utils.io import get_station_paths


def main():
    parser = argparse.ArgumentParser(description="CRNP 전처리 테스트")
    parser.add_argument("--station", "-s", required=True, help="관측소 ID (예: HC)")
    args = parser.parse_args()

    station_id = args.station

    # ── 경로 확인 ─────────────────────────────────────────────────────────
    paths = get_station_paths(station_id)
    raw_crnp  = paths["raw_crnp"]
    processed = paths["processed"]

    print(f"\n{'='*55}")
    print(f"  CRNP 전처리 테스트  —  {station_id}")
    print(f"{'='*55}")
    print(f"  입력 폴더 : {raw_crnp}")
    print(f"  출력 폴더 : {processed}")

    # .dat 파일 목록 확인
    dat_files = sorted(raw_crnp.glob("*.dat"))
    if not dat_files:
        print(f"\n❌ .dat 파일 없음: {raw_crnp}")
        sys.exit(1)

    print(f"\n  발견된 .dat 파일 ({len(dat_files)}개):")
    for f in dat_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"    - {f.name}  ({size_mb:.1f} MB)")

    # ── 전처리 실행 ────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    proc = CRNPProcessor(station_id, input_dir=raw_crnp, output_dir=processed)
    out_path = proc.process()

    # ── 결과 상세 확인 ─────────────────────────────────────────────────────
    import pandas as pd
    import numpy as np

    df = pd.read_excel(out_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(f"\n{'─'*55}")
    print(f"  결과 미리보기 (처음 5행)")
    print(f"{'─'*55}")
    print(df.head().to_string(index=False))

    print(f"\n{'─'*55}")
    print(f"  결과 미리보기 (마지막 5행)")
    print(f"{'─'*55}")
    print(df.tail().to_string(index=False))

    print(f"\n{'─'*55}")
    print(f"  컬럼별 결측값")
    print(f"{'─'*55}")
    for col in df.columns:
        if col == "timestamp":
            continue
        n_nan  = df[col].isna().sum()
        pct    = n_nan / len(df) * 100
        marker = " ⚠️" if pct > 5 else ""
        print(f"  {col:<15}: {n_nan:>5}개 ({pct:.1f}%){marker}")

    print(f"\n{'─'*55}")
    print(f"  N_counts 분포 확인")
    print(f"{'─'*55}")
    nc = df["N_counts"].dropna()
    print(f"  min   : {nc.min():.0f}")
    print(f"  Q1    : {nc.quantile(0.25):.0f}")
    print(f"  median: {nc.median():.0f}")
    print(f"  Q3    : {nc.quantile(0.75):.0f}")
    print(f"  max   : {nc.max():.0f}")
    print(f"  mean  : {nc.mean():.1f}")
    print(f"  std   : {nc.std():.1f}")

    # 낮은 값 이상치 확인 + 날짜 출력
    low_threshold = nc.mean() - 3 * nc.std()
    low_mask = df["N_counts"] < low_threshold
    n_low = low_mask.sum()
    if n_low > 0:
        print(f"  ⚠️  평균-3σ ({low_threshold:.0f}) 미만: {n_low}개")
        print(f"      해당 시각:")
        low_rows = df[low_mask][["timestamp", "N_counts", "Ta", "RH", "Pa"]]
        for _, row in low_rows.iterrows():
            print(f"      {row['timestamp']}  N={row['N_counts']:.0f}  "
                  f"Ta={row['Ta']:.1f}°C  RH={row['RH']:.1f}%")

    print(f"\n{'─'*55}")
    print(f"  월별 데이터 수")
    print(f"{'─'*55}")
    monthly = df.groupby(df["timestamp"].dt.to_period("M")).size()
    for period, count in monthly.items():
        expected = pd.Period(period).days_in_month * 24
        completeness = count / expected * 100
        bar = "█" * int(completeness / 5)
        print(f"  {period}  {count:>4}행  {completeness:>5.1f}%  {bar}")

    print(f"\n✅ 저장 완료: {out_path}\n")


if __name__ == "__main__":
    main()