# src/calibration/base.py
"""
BaseCalibrator: 모든 교정 방법의 공통 인터페이스.

[분석 기간 vs 교정 기간]
  analysis_start/end    : VWC를 산출할 전체 기간
                          → run_calibration.py에서 matched_df를 이 범위로 필터링
  calibration_start/end : 교정 파라미터(N0, a2, ND)를 결정하는 기간
                          → 각 캘리브레이터 내부에서 matched_df의 부분집합으로 사용

  흐름:
    matched_df (analysis 기간 전체)
        ├─ cal_df  (calibration 기간 필터) → 파라미터 결정
        └─ eval_df (exclude_months만 제외) → VWC 계산 및 평가

  우선순위 (get_cal_period):
    1. 방법별 calibration_start/end
    2. 공통 calibration_start/end
    3. null → matched_df 전체 범위 사용
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# JSON 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _save_json(data: Dict, path: Path) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):   return int(obj)
        if isinstance(obj, (np.floating,)):  return float(obj)
        if isinstance(obj, np.ndarray):      return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_convert)


def _load_json(path: Path) -> Optional[Dict]:
    import json
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 메트릭 계산
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    valid = np.isfinite(obs) & np.isfinite(pred)
    o, p  = obs[valid], pred[valid]
    n     = int(len(o))
    if n < 2:
        return dict(RMSE=np.nan, MAE=np.nan, Bias=np.nan,
                    R=np.nan, NSE=np.nan, n=n)
    rmse  = float(np.sqrt(np.mean((o - p) ** 2)))
    mae   = float(np.mean(np.abs(o - p)))
    bias  = float(np.mean(p - o))
    r     = float(np.corrcoef(o, p)[0, 1])
    ss_r  = np.sum((o - p) ** 2)
    ss_t  = np.sum((o - np.mean(o)) ** 2)
    nse   = float(1 - ss_r / ss_t) if ss_t > 0 else np.nan
    return dict(RMSE=rmse, MAE=mae, Bias=bias, R=r, NSE=nse, n=n)


# ─────────────────────────────────────────────────────────────────────────────
# CalibrationResult
# ─────────────────────────────────────────────────────────────────────────────

class CalibrationResult:
    """교정 결과 컨테이너."""

    def __init__(self,
                 method:      str,
                 station_id:  str,
                 N0:          float,
                 a2:          float,
                 vwc:         np.ndarray,
                 obs:         np.ndarray,
                 cal_date:    Optional[Any] = None,
                 extra:       Optional[Dict] = None):
        self.method     = method
        self.station_id = station_id
        self.N0         = float(N0)
        self.a2         = float(a2)
        self.vwc        = vwc
        self.obs        = obs
        self.cal_date   = str(cal_date) if cal_date is not None else None
        self.metrics    = _compute_metrics(obs, vwc)
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

    def _filename(self) -> str:
        return f"{self.station_id}_{self.method}_calibration_result.json"

    def save(self, result_dir: Path) -> Path:
        path = result_dir / self.station_id / self.method / self._filename()
        _save_json(self.to_dict(), path)
        return path

    @classmethod
    def load(cls, result_dir: Path, station_id: str, method: str) -> Optional[Dict]:
        fname = f"{station_id}_{method}_calibration_result.json"
        path  = result_dir / station_id / method / fname
        return _load_json(path)

    def summary(self) -> str:
        m = self.metrics
        date_str = f"  cal_date={self.cal_date}" if self.cal_date else ""
        return (f"[{self.method:<12}] "
                f"N0={self.N0:7.1f}  a2={self.a2:.4f}  "
                f"RMSE={m['RMSE']:.4f}  R={m['R']:.3f}  n={m['n']}"
                f"{date_str}")


# ─────────────────────────────────────────────────────────────────────────────
# BaseCalibrator
# ─────────────────────────────────────────────────────────────────────────────

class BaseCalibrator(ABC):

    A0: float = 0.0808
    A1: float = 0.372

    def __init__(self, station_cfg: Dict, options: Dict):
        self.station_config = station_cfg
        self.options        = options
        self.station_id     = station_cfg["station_info"]["id"]

        soil = station_cfg.get("soil_properties", {})
        self.bulk_density  = float(soil.get("bulk_density", 1.44))
        self.clay_content  = float(soil.get("clay_content", 0.35))
        lw = soil.get("lattice_water")
        self.lattice_water: Optional[float] = float(lw) if lw is not None else None

        self.result: Optional[CalibrationResult] = None

    @abstractmethod
    def calibrate(self, matched_df: pd.DataFrame) -> CalibrationResult:
        pass

    # ── 교정 기간 헬퍼 ────────────────────────────────────────────────────────

    def get_cal_period(self, method_key: str) -> Tuple[Optional[str], Optional[str]]:
        """
        교정 기간(calibration_start/end) 결정.

        우선순위:
          1. YAML calibration.<method_key>.calibration_start/end
          2. YAML calibration.calibration_start/end  (공통값)
          3. None → matched_df 전체 사용

        Parameters
        ----------
        method_key : str  예: "standard", "shp_joint", "shp_2pt", "uts"
        """
        cal  = self.options.get("calibration", {})
        meth = cal.get(method_key, {})

        # 방법별 override → 공통값 → None 순서
        start = (meth.get("calibration_start")
                 or cal.get("calibration_start"))
        end   = (meth.get("calibration_end")
                 or cal.get("calibration_end"))
        return start, end

    def filter_cal(self,
                   df:             pd.DataFrame,
                   cal_start:      Optional[str],
                   cal_end:        Optional[str],
                   exclude_months: List[int],
                   ref_col:        str) -> pd.DataFrame:
        """
        matched_df에서 교정 기간 + 제외월 필터링.
        파라미터 결정에 사용할 cal_df 반환.
        """
        mask = pd.Series(True, index=df.index)
        if cal_start:
            mask &= df["date"] >= pd.to_datetime(cal_start)
        if cal_end:
            mask &= df["date"] <= pd.to_datetime(cal_end)
        mask &= ~df["date"].dt.month.isin(exclude_months)
        mask &= df[ref_col].notna()
        mask &= df["N_corrected"].notna()
        return df[mask].copy()

    def filter_eval(self,
                    df:             pd.DataFrame,
                    exclude_months: List[int],
                    ref_col:        str) -> pd.DataFrame:
        """
        matched_df 전체에서 제외월만 필터링.
        VWC 계산 및 평가에 사용할 eval_df 반환.
        """
        mask = (
            ~df["date"].dt.month.isin(exclude_months)
            & df[ref_col].notna()
            & df["N_corrected"].notna()
        )
        return df[mask].copy()

    # ── 기타 헬퍼 ─────────────────────────────────────────────────────────────

    @staticmethod
    def vwc_from_N(N: np.ndarray, N0: float, a2: float) -> np.ndarray:
        denom = N / N0 - BaseCalibrator.A1
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denom > 0, BaseCalibrator.A0 / denom - a2, np.nan)

    def save(self, result_dir: Path) -> None:
        if self.result is None:
            raise RuntimeError("calibrate() 를 먼저 실행하세요.")
        self.result.save(result_dir)

    def summary(self) -> str:
        if self.result is None:
            return f"[{self.__class__.__name__}] Not calibrated yet."
        return self.result.summary()