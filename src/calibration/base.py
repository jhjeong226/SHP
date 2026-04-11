# src/calibration/base.py
"""
BaseCalibrator: 모든 교정 방법의 공통 인터페이스.

calibrate(matched_df, eval_df) 를 하위 클래스가 구현.
  matched_df : date | theta | N   (전체 기간 FDR-CRNP 일별 매칭)
  eval_df    : date | FDR_avg     (10cm 일평균, 성능 평가 기준)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..utils.io import save_json, load_json
from ..utils.metrics import compute_metrics


class CalibrationResult:
    """교정 결과 컨테이너 — 모든 방법이 동일한 구조로 반환"""

    def __init__(self,
                 method:      str,
                 station_id:  str,
                 N0:          float,
                 a2:          float,
                 vwc:         np.ndarray,   # 평가 기간 VWC 예측값
                 obs:         np.ndarray,   # 평가 기간 FDR_avg 관측값
                 cal_date:    Optional[Any] = None,  # 교정 기준 날짜
                 extra:       Optional[Dict] = None):
        self.method     = method
        self.station_id = station_id
        self.N0         = float(N0)
        self.a2         = float(a2)
        self.vwc        = vwc
        self.obs        = obs
        self.cal_date   = str(cal_date) if cal_date is not None else None
        self.metrics    = compute_metrics(obs, vwc)
        self.extra      = extra or {}
        self.timestamp  = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return dict(
            method=self.method,
            station_id=self.station_id,
            N0=self.N0,
            a2=self.a2,
            cal_date=self.cal_date,
            metrics=self.metrics,
            extra=self.extra,
            timestamp=self.timestamp,
        )

    def save(self, result_dir: Path) -> None:
        path = result_dir / self.station_id / self.method / "calibration_result.json"
        save_json(self.to_dict(), path)

    @classmethod
    def load(cls, result_dir: Path, station_id: str, method: str) -> Optional[Dict]:
        path = result_dir / station_id / method / "calibration_result.json"
        return load_json(path)

    def summary(self) -> str:
        m = self.metrics
        date_str = f"  cal_date={self.cal_date}" if self.cal_date else ""
        return (f"[{self.method:<12}] "
                f"N0={self.N0:7.1f}  a2={self.a2:.4f}  "
                f"RMSE={m['RMSE']:.4f}  R={m['R']:.3f}  n={m['n']}"
                f"{date_str}")


# ════════════════════════════════════════════════════════════════════════════

class BaseCalibrator(ABC):
    """
    모든 교정 클래스의 추상 기반.

    Parameters
    ----------
    station_config : dict  (YAML station 섹션)
    options        : dict  (YAML processing_options 섹션)
    """

    A0: float = 0.0808   # 고정 물리 상수 (Desilets et al.)
    A1: float = 0.372

    def __init__(self, station_config: Dict, options: Dict):
        self.station_config = station_config
        self.options        = options
        self.station_id     = station_config["station_info"]["id"]

        soil = station_config.get("soil_properties", {})
        self.bulk_density  = float(soil.get("bulk_density", 1.44))
        self.clay_content  = float(soil.get("clay_content", 0.35))
        lw = soil.get("lattice_water")
        self.lattice_water: Optional[float] = float(lw) if lw is not None else None

        self.result: Optional[CalibrationResult] = None

    # ── 하위 클래스가 반드시 구현 ──────────────────────────────────────────
    @abstractmethod
    def calibrate(self,
                  matched_df: pd.DataFrame,
                  eval_df:    pd.DataFrame,
                  **kwargs) -> CalibrationResult:
        """
        교정 수행.

        Parameters
        ----------
        matched_df : columns [date, theta, N]
            전체 기간의 일별 (FDR 가중평균, CRNP 보정 중성자) 쌍
        eval_df    : columns [date, FDR_avg]
            10cm 일평균 — 성능 평가 기준 (4~10월)
        """

    # ── 공통 헬퍼 ──────────────────────────────────────────────────────────
    @staticmethod
    def vwc_from_N(N: np.ndarray, N0: float, a2: float) -> np.ndarray:
        denom = N / N0 - BaseCalibrator.A1
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denom > 0, BaseCalibrator.A0 / denom - a2, np.nan)

    def _auto_lattice_water(self) -> float:
        if self.lattice_water is not None:
            return self.lattice_water
        try:
            import crnpy
            lw = float(crnpy.lattice_water(clay_content=self.clay_content))
        except Exception:
            lw = 0.02 + 0.5 * self.clay_content * 0.1
        self.lattice_water = lw
        return lw

    def _predict_vwc(self, N: np.ndarray, N0: float, wlat: float) -> np.ndarray:
        try:
            import crnpy
            vwc = crnpy.counts_to_vwc(N, N0=N0,
                                       bulk_density=self.bulk_density,
                                       Wlat=wlat, Wsoc=0.01)
        except Exception:
            vwc = self.vwc_from_N(N, N0, a2=wlat)
        return np.clip(np.asarray(vwc, dtype=float), 0, 1)

    def save(self, result_dir: Path) -> None:
        if self.result is None:
            raise RuntimeError("calibrate() 를 먼저 실행하세요.")
        self.result.save(result_dir)

    def summary(self) -> str:
        if self.result is None:
            return f"[{self.__class__.__name__}] Not calibrated yet."
        return self.result.summary()