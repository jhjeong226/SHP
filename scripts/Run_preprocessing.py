#!/usr/bin/env python
# scripts/run_preprocessing.py
"""
전처리 실행 스크립트
====================
FDR (3단계) + CRNP (보정 + 일자료) 전처리를 순서대로 실행.

사용법:
  python scripts/run_preprocessing.py --station HC
  python scripts/run_preprocessing.py --station HC PC SW
  python scripts/run_preprocessing.py --station HC --skip-crnp
  python scripts/run_preprocessing.py --station HC --skip-fdr
"""

import argparse
import sys
import traceback
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.fdr  import FDRProcessor
from src.preprocessing.crnp import CRNPProcessor
from src.utils.io import load_config, get_station_paths


# ═══════════════════════════════════════════════════════════════════════════════
# 단일 관측소 처리
# ═══════════════════════════════════════════════════════════════════════════════

def run_fdr(station_id: str, cfg: dict, paths: dict) -> bool:
    """FDR 전처리 실행. 성공 여부 반환."""
    try:
        proc = FDRProcessor(
            station_id  = station_id,
            station_cfg = cfg["station"],
            options     = cfg["options"],
            input_dir   = paths["raw_fdr"],
            output_dir  = paths["processed"],
        )
        result = proc.run()

        print("\n  FDR 출력 파일:")
        print(f"    [시간] {result['hourly'].name}")
        print(f"    [일별] {result['daily'].name}")
        return True

    except FileNotFoundError as e:
        print(f"\n  ❌ FDR 입력 없음: {e}")
        return False
    except Exception as e:
        print(f"\n  ❌ FDR 전처리 실패: {e}")
        traceback.print_exc()
        return False


def run_crnp(station_id: str, cfg: dict, paths: dict) -> bool:
    """CRNP 전처리 실행. 성공 여부 반환."""
    try:
        proc = CRNPProcessor(
            station_id  = station_id,
            station_cfg = cfg["station"],
            options     = cfg["options"],
            input_dir   = paths["raw_crnp"],
            output_dir  = paths["processed"],
        )
        out_path = proc.run()
        print(f"\n  CRNP 출력 파일: {out_path.name}")
        return True

    except FileNotFoundError as e:
        print(f"\n  ❌ CRNP 입력 없음: {e}")
        return False
    except Exception as e:
        print(f"\n  ❌ CRNP 전처리 실패: {e}")
        traceback.print_exc()
        return False


def process_station(station_id: str,
                    skip_fdr:  bool = False,
                    skip_crnp: bool = False) -> None:
    """단일 관측소 전처리."""
    print(f"\n{'#'*65}")
    print(f"  관측소: {station_id}")
    print(f"{'#'*65}")

    try:
        cfg   = load_config(station_id)
        paths = get_station_paths(station_id)
    except FileNotFoundError as e:
        print(f"  ❌ 설정 파일 오류: {e}")
        return

    results = {}

    if not skip_fdr:
        print(f"\n▶ FDR 전처리 시작  ({station_id})")
        results["fdr"] = run_fdr(station_id, cfg, paths)
    else:
        print(f"\n  (FDR 건너뜀)")

    if not skip_crnp:
        print(f"\n▶ CRNP 전처리 시작  ({station_id})")
        results["crnp"] = run_crnp(station_id, cfg, paths)
    else:
        print(f"\n  (CRNP 건너뜀)")

    # 결과 요약
    print(f"\n{'─'*65}")
    print(f"  [{station_id}] 전처리 결과 요약")
    for key, ok in results.items():
        status = "✅ 성공" if ok else "❌ 실패"
        print(f"    {key.upper():<6} : {status}")
    print(f"{'─'*65}")


# ═══════════════════════════════════════════════════════════════════════════════
# 진입점
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FDR + CRNP 전처리 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python scripts/run_preprocessing.py --station HC
  python scripts/run_preprocessing.py --station HC PC SW
  python scripts/run_preprocessing.py --station HC --skip-crnp
        """,
    )
    parser.add_argument(
        "--station", "-s",
        nargs="+",
        required=True,
        help="관측소 ID (예: HC  또는  HC PC SW)",
    )
    parser.add_argument(
        "--skip-fdr",
        action="store_true",
        help="FDR 전처리 건너뜀",
    )
    parser.add_argument(
        "--skip-crnp",
        action="store_true",
        help="CRNP 전처리 건너뜀",
    )
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  CRNP SHP Calibration System — 전처리 실행")
    print(f"  대상 관측소: {args.station}")
    print(f"{'='*65}")

    for station_id in args.station:
        process_station(
            station_id = station_id.upper(),
            skip_fdr   = args.skip_fdr,
            skip_crnp  = args.skip_crnp,
        )

    print(f"\n✅ 전체 전처리 완료\n")


if __name__ == "__main__":
    main()