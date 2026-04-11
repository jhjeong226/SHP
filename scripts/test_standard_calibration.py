# scripts/test_standard_calibration.py
"""
Standard 교정 테스트 스크립트 (Desilet 방식)
실행: python scripts/test_standard_calibration.py --station HC
"""

import argparse, sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.utils.io import load_config, get_station_paths
from src.calibration.correction import NeutronCorrector
from src.calibration.matching   import DataMatcher
from src.calibration.standard   import StandardCalibrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", "-s", required=True)
    parser.add_argument("--match_start", default=None,
                        help="매칭 시작일 YYYY-MM-DD (기본: FDR 시작일)")
    parser.add_argument("--match_end", default=None,
                        help="매칭 종료일 YYYY-MM-DD (기본: FDR 종료일)")
    args = parser.parse_args()

    station_id  = args.station
    cfg         = load_config(station_id)
    paths       = get_station_paths(station_id)
    station_cfg = cfg["station"]
    options     = cfg["options"]
    processed   = paths["processed"]

    print(f"\n{'='*60}")
    print(f"  Standard 교정 (Desilet 방식)  —  {station_id}")
    print(f"{'='*60}")

    # ── 1. 데이터 로드 ────────────────────────────────────────────────────────
    print(f"\n▶ [1] 데이터 로드")
    crnp_df  = pd.read_excel(processed / f"{station_id}_CRNP_hourly.xlsx")
    fdr_df   = pd.read_csv(processed / f"{station_id}_FDR_hourly.csv", low_memory=False)
    daily_avg = pd.read_excel(processed / f"{station_id}_FDR_daily_avg.xlsx")

    for df in [crnp_df, fdr_df]:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    daily_avg["date"] = pd.to_datetime(daily_avg["Date"])
    daily_avg = daily_avg.rename(columns={"10cm": "FDR_avg"})[["date", "FDR_avg"]]

    print(f"  CRNP    : {len(crnp_df):,}행  "
          f"({crnp_df['timestamp'].min().date()} ~ {crnp_df['timestamp'].max().date()})")
    print(f"  FDR     : {len(fdr_df):,}행  "
          f"({fdr_df['timestamp'].min().date()} ~ {fdr_df['timestamp'].max().date()})")
    print(f"  FDR_avg : {len(daily_avg):,}일  "
          f"({daily_avg['date'].min().date()} ~ {daily_avg['date'].max().date()})")

    # ── 2. 중성자 보정 ────────────────────────────────────────────────────────
    # 참조값은 FDR-CRNP 겹치는 전체 기간의 평균 사용 (원본 코드와 동일)
    print(f"\n▶ [2] 중성자 보정")
    corrector   = NeutronCorrector(station_cfg, options)
    overlap_start = max(crnp_df["timestamp"].min(), fdr_df["timestamp"].min())
    overlap_end   = min(crnp_df["timestamp"].max(), fdr_df["timestamp"].max())
    refs = corrector.reference_values(crnp_df,
                                      str(overlap_start.date()),
                                      str(overlap_end.date()))
    print(f"  참조값 (FDR-CRNP 겹치는 기간 평균): "
          f"Pref={refs.get('Pref', 0):.2f}  Aref={refs.get('Aref', 0):.4f}")
    corrected = corrector.correct(crnp_df,
                                  Pref=refs.get("Pref"),
                                  Aref=refs.get("Aref"))

    # ── 3. 일별 매칭 (전체 겹치는 기간) ──────────────────────────────────────
    print(f"\n▶ [3] FDR-CRNP 일별 매칭 (전체 기간)")
    match_start = args.match_start or str(overlap_start.date())
    match_end   = args.match_end   or str(overlap_end.date())
    print(f"  기간: {match_start} ~ {match_end}")

    matcher    = DataMatcher(station_cfg)
    matched_df = matcher.match(fdr_df, corrected, match_start, match_end)

    if matched_df.empty:
        print("❌ 매칭 실패"); sys.exit(1)

    # ── 4. Standard 교정 ──────────────────────────────────────────────────────
    print(f"\n▶ [4] Standard 교정 (날짜 탐색)")
    calibrator = StandardCalibrator(station_cfg, options)
    result     = calibrator.calibrate(matched_df, daily_avg)

    # ── 5. 결과 저장 ──────────────────────────────────────────────────────────
    results_dir = project_root / "results"
    result.save(results_dir)

    # ── 6. 요약 출력 ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  {result.summary()}")
    m = result.metrics
    print(f"  MAE  = {m['MAE']:.4f}")
    print(f"  Bias = {m['Bias']:.4f}")
    print(f"  NSE  = {m['NSE']:.4f}")

    # ── 7. 시각화 ─────────────────────────────────────────────────────────────
    eval_df = daily_avg.copy()
    eval_df["date"] = pd.to_datetime(eval_df["date"])

    matched_eval = matched_df[
        matched_df["date"].apply(lambda d: pd.Timestamp(d).month) in range(4, 11)
        if False else
        matched_df["date"].apply(lambda d: pd.Timestamp(d).month >= 4 and
                                           pd.Timestamp(d).month <= 10)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{station_id} — Standard Calibration  "
                 f"(cal_date: {result.cal_date})", fontsize=13)

    # 시계열
    ax = axes[0]
    dates = pd.to_datetime(eval_df["date"])
    ax.plot(dates, eval_df["FDR_avg"], "k-", lw=1.2, label="FDR_avg (10cm)")
    ax.plot(dates, result.vwc,         "r-", lw=1.2,
            label=f"Standard  RMSE={m['RMSE']:.4f}")
    ax.set_xlabel("Date"); ax.set_ylabel("VWC (m³/m³)")
    ax.set_title("Time Series (4~10월 평가 기간)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45)

    # 산포도
    ax = axes[1]
    obs, pred = result.obs, result.vwc
    valid = np.isfinite(obs) & np.isfinite(pred)
    lim = (min(obs[valid].min(), pred[valid].min()) - 0.01,
           max(obs[valid].max(), pred[valid].max()) + 0.01)
    ax.plot(lim, lim, "k--", lw=1)
    ax.scatter(obs[valid], pred[valid], s=25, color="#4477AA",
               edgecolors="k", lw=0.4, alpha=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("FDR_avg (10cm)"); ax.set_ylabel("CRNP VWC")
    ax.set_title(f"Scatter  R={m['R']:.3f}  n={m['n']}")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_png = results_dir / station_id / "standard" / "calibration.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  그래프: {out_png}")
    print()


if __name__ == "__main__":
    main()