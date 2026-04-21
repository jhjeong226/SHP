#!/usr/bin/env python
# scripts/run_calibration.py
"""
교정 실행 스크립트
==================
DataMatcher → 방법별 Calibrator 순서로 실행.
실행할 방법은 processing_options.yaml의 calibration.methods 로 제어.

사용법:
  python scripts/run_calibration.py --station HC
  python scripts/run_calibration.py --station HC PC SW
"""

import argparse
import pandas as pd
import sys
import traceback
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.calibration.matcher  import DataMatcher
from src.calibration.standard import StandardCalibrator
from src.calibration.shp_2pt  import SHP2ptCalibrator
from src.calibration.shp_joint import SHPJointCalibrator
from src.calibration.uts       import UTSCalibrator
from src.calibration.shp_opt   import SHPOptCalibrator
from src.utils.io import load_config, get_station_paths


# ─────────────────────────────────────────────────────────────────────────────
# 방법 레지스트리 (이름 → 클래스)
# ─────────────────────────────────────────────────────────────────────────────
METHOD_REGISTRY = {
    "standard":  StandardCalibrator,
    "shp_joint": SHPJointCalibrator,
    "shp_2pt":   SHP2ptCalibrator,
    "uts":        UTSCalibrator,
    "shp_opt":    SHPOptCalibrator,
}


# ─────────────────────────────────────────────────────────────────────────────
# 단일 관측소 처리
# ─────────────────────────────────────────────────────────────────────────────

def run_station(station_id: str) -> bool:
    try:
        cfg   = load_config(station_id)
        paths = get_station_paths(station_id)

        station_cfg = cfg["station"]
        options     = cfg["options"]
        methods     = options.get("calibration", {}).get("methods", ["standard"])

        processed_dir = paths["processed"]
        result_dir    = Path(options.get("data_root", "data")).parent / "results"
        result_dir.mkdir(parents=True, exist_ok=True)

        # ── DataMatcher (모든 방법 공통) ─────────────────────────────────
        print(f"\n{'▶'*3} DataMatcher")
        matcher    = DataMatcher(station_cfg=station_cfg,
                                 options=options,
                                 processed_dir=processed_dir)
        matched_df = matcher.run()

        # ── 분석 기간 필터링 (VWC 산출 전체 범위) ───────────────────────
        cal_opts       = options.get("calibration", {})
        analysis_start = cal_opts.get("analysis_start", None)
        analysis_end   = cal_opts.get("analysis_end",   None)

        matched_df["date"] = pd.to_datetime(matched_df["date"])
        if analysis_start or analysis_end:
            before = len(matched_df)
            if analysis_start:
                matched_df = matched_df[
                    matched_df["date"] >= pd.to_datetime(analysis_start)
                ]
            if analysis_end:
                matched_df = matched_df[
                    matched_df["date"] <= pd.to_datetime(analysis_end)
                ]
            matched_df = matched_df.reset_index(drop=True)
            print(f"\n  📅 분석 기간 (VWC 출력): {analysis_start or '전체'} ~ "
                  f"{analysis_end or '전체'}  ({before}일 → {len(matched_df)}일)")

        # 교정 기간 로그 출력 (실제 필터링은 각 캘리브레이터 내부에서)
        cal_start = cal_opts.get("calibration_start", None)
        cal_end   = cal_opts.get("calibration_end",   None)
        if cal_start or cal_end:
            print(f"  🔧 교정 기간 (파라미터 결정): "
                  f"{cal_start or '전체'} ~ {cal_end or '전체'}")

        # ── 방법별 교정 ──────────────────────────────────────────────────
        results = {}
        for method_name in methods:
            CalClass = METHOD_REGISTRY.get(method_name)
            if CalClass is None:
                print(f"\n  ⚠️  알 수 없는 방법: {method_name} (건너뜀)")
                continue

            print(f"\n{'▶'*3} {method_name}")
            try:
                cal    = CalClass(station_cfg=station_cfg, options=options)
                result = cal.calibrate(matched_df)

                print(f"\n{'─'*60}")
                print(result.summary())
                print(f"{'─'*60}")

                cal.save_result(result_dir)

                print(f"\n{'▶'*3} [시각화] {method_name}")
                cal.plot_result(matched_df, result_dir)

                results[method_name] = result

            except Exception as e:
                print(f"\n  ❌ [{method_name}] 실패: {e}")
                traceback.print_exc()

        # ── 방법 간 요약 비교 ────────────────────────────────────────────
        if len(results) > 1:
            print(f"\n{'='*60}")
            print(f"  [{station_id}] 방법 비교")
            print(f"{'─'*60}")
            print(f"  {'방법':<14} {'N0':>8} {'a2':>8} {'RMSE':>8} {'R':>7} {'NSE':>7}")
            print(f"  {'─'*55}")
            for name, res in results.items():
                m = res.metrics
                print(f"  {name:<14} {res.N0:8.1f} {res.a2:8.4f} "
                      f"{m['RMSE']:8.4f} {m['R']:7.3f} {m['NSE']:7.3f}")
            print(f"{'='*60}")

        return True

    except FileNotFoundError as e:
        print(f"\n  ❌ 파일 없음: {e}")
        print(f"     전처리(run_preprocessing.py)를 먼저 실행하세요.")
        return False
    except Exception as e:
        print(f"\n  ❌ 교정 실패: {e}")
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="N0 교정 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python scripts/run_calibration.py --station HC
  python scripts/run_calibration.py --station HC PC SW
        """,
    )
    parser.add_argument("--station", "-s", nargs="+", required=True)
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  CRNP SHP Calibration System — N0 교정 실행")
    print(f"  대상 관측소: {args.station}")
    print(f"{'='*65}")

    summary = {}
    for station_id in args.station:
        sid = station_id.upper()
        print(f"\n{'#'*65}")
        print(f"  관측소: {sid}")
        print(f"{'#'*65}")
        summary[sid] = run_station(sid)

    print(f"\n{'='*65}")
    print("  전체 교정 요약")
    print(f"{'─'*65}")
    for sid, ok in summary.items():
        print(f"  {sid:<6} : {'✅ 성공' if ok else '❌ 실패'}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()