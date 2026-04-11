# scripts/debug_matching.py
"""
매칭 데이터 진단 스크립트
실행: python scripts/debug_matching.py --station HC
"""

import argparse, sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from src.utils.io import load_config, get_station_paths
from src.calibration.correction import NeutronCorrector
from src.calibration.matching   import DataMatcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", "-s", required=True)
    args = parser.parse_args()

    station_id  = args.station
    cfg         = load_config(station_id)
    paths       = get_station_paths(station_id)
    station_cfg = cfg["station"]
    options     = cfg["options"]

    cal_cfg = station_cfg["calibration"]
    start   = cal_cfg["default_start_date"]
    end     = cal_cfg["default_end_date"]

    processed = paths["processed"]
    crnp_df = pd.read_excel(processed / f"{station_id}_CRNP_hourly.xlsx")
    fdr_df  = pd.read_csv(processed / f"{station_id}_FDR_hourly.csv", low_memory=False)
    crnp_df["timestamp"] = pd.to_datetime(crnp_df["timestamp"])
    fdr_df["timestamp"]  = pd.to_datetime(fdr_df["timestamp"])

    # ── fi 끄고 보정 ──────────────────────────────────────────────────────────
    options["corrections"]["incoming_flux"] = False
    corrector = NeutronCorrector(station_cfg, options)
    refs      = corrector.reference_values(crnp_df, start, end)
    corrected = corrector.correct(crnp_df, Pref=refs.get("Pref"), Aref=refs.get("Aref"))

    # ── 매칭 ─────────────────────────────────────────────────────────────────
    matcher = DataMatcher(station_cfg)
    matched = matcher.match(fdr_df, corrected, start, end)

    print(f"\n{'='*60}")
    print(f"  매칭 데이터 상세  ({start} ~ {end})")
    print(f"{'='*60}")
    print(f"  {'날짜':<12} {'θ(FDR)':>8} {'N_corr':>9} {'n_sensors':>10}")
    print(f"  {'-'*45}")
    for _, row in matched.iterrows():
        print(f"  {str(row['date']):<12} {row['theta']:>8.4f} "
              f"{row['N']:>9.1f} {int(row['n_sensors']):>10}")

    # ── 상관관계 진단 ─────────────────────────────────────────────────────────
    theta = matched["theta"].values
    N     = matched["N"].values
    r     = float(np.corrcoef(theta, N)[0, 1])

    print(f"\n  Pearson R (θ vs N) = {r:.3f}")
    if r > 0:
        print("  ⚠️  양의 상관 → N 증가 시 θ 증가 (물리적으로 반대여야 함)")
        print("     원인 후보: FDR 가중평균 오류 또는 매칭 날짜 불일치")
    else:
        print("  ✅ 음의 상관 (N 증가 → θ 감소) — 정상")

    # ── FDR 단순 평균으로 재확인 ─────────────────────────────────────────────
    fdr_s   = fdr_df.copy()
    fdr_s["date"] = fdr_s["timestamp"].dt.date
    crnp_s  = corrected.copy()
    crnp_s["date"] = crnp_s["timestamp"].dt.date

    daily_fdr  = (fdr_s[(fdr_s["timestamp"] >= pd.to_datetime(start)) &
                         (fdr_s["timestamp"] <= pd.to_datetime(end))]
                  .groupby("date")["theta_v"].mean())
    daily_crnp = (crnp_s[(crnp_s["timestamp"] >= pd.to_datetime(start)) &
                          (crnp_s["timestamp"] <= pd.to_datetime(end))]
                  .groupby("date")["N_corrected"].mean())
    common = daily_fdr.index.intersection(daily_crnp.index)
    r_simple = float(np.corrcoef(daily_fdr[common], daily_crnp[common])[0, 1])

    print(f"\n  단순평균 FDR vs N_corrected R = {r_simple:.3f}")
    if abs(r_simple) > abs(r):
        print("  → 가중평균 문제 의심")
    else:
        print("  → 가중평균과 유사, 근본적 데이터 문제 의심")

    # ── FDR 깊이별 상관관계 ──────────────────────────────────────────────────
    print(f"\n  깊이별 FDR vs N 상관관계:")
    for depth in sorted(fdr_s["depth_cm"].unique()):
        sub = (fdr_s[(fdr_s["depth_cm"] == depth) &
                     (fdr_s["timestamp"] >= pd.to_datetime(start)) &
                     (fdr_s["timestamp"] <= pd.to_datetime(end))]
               .groupby("date")["theta_v"].mean())
        common_d = sub.index.intersection(daily_crnp.index)
        if len(common_d) > 3:
            r_d = float(np.corrcoef(sub[common_d], daily_crnp[common_d])[0, 1])
            print(f"    {depth}cm: R = {r_d:.3f}")

    # ── 사이트별 상관관계 ────────────────────────────────────────────────────
    print(f"\n  사이트별 FDR 10cm vs N 상관관계:")
    for site in sorted(fdr_s["site_id"].unique()):
        sub = (fdr_s[(fdr_s["site_id"] == site) &
                     (fdr_s["depth_cm"] == fdr_s["depth_cm"].min()) &
                     (fdr_s["timestamp"] >= pd.to_datetime(start)) &
                     (fdr_s["timestamp"] <= pd.to_datetime(end))]
               .groupby("date")["theta_v"].mean())
        common_s = sub.index.intersection(daily_crnp.index)
        if len(common_s) > 3:
            r_s = float(np.corrcoef(sub[common_s], daily_crnp[common_s])[0, 1])
            print(f"    {site}: R = {r_s:.3f}")


if __name__ == "__main__":
    main()