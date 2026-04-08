# src/calibration/base.py
"""
BaseCalibrator: 모든 교정 방법의 공통 인터페이스.

하위 클래스는 calibrate() 만 구현하면 됩니다.
저장/로드/요약은 공통으로 처리됩니다.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..utils.io import save_json, load_json
from ..utils.metrics import compute_metrics


class CalibrationResult:
    """교정 결과 컨테이너 — 모든 방법이 동일한 구조로 반환"""

    def __init__(self,
                 method: str,
                 station_id: str,
                 N0: float,
                 a2: float,
                 vwc: np.ndarray,
                 obs: np.ndarray,
                 extra: Optional[Dict] = None):
        self.method     = method
        self.station_id = station_id
        self.N0         = float(N0)
        self.a2         = float(a2)
        self.vwc        = vwc              # 교정 기간 VWC 예측값
        self.obs        = obs              # 교정 기간 FDR 관측값
        self.metrics    = compute_metrics(obs, vwc)
        self.extra      = extra or {}      # 방법별 추가 정보
        self.timestamp  = datetime.now().isoformat()

    # ── 직렬화 ─────────────────────────────────────────────────────────────
    def to_dict(self) -> Dict:
        return dict(
            method=self.method,
            station_id=self.station_id,
            N0=self.N0,
            a2=self.a2,
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

    # ── 요약 출력 ──────────────────────────────────────────────────────────
    def summary(self) -> str:
        m = self.metrics
        return (f"[{self.method:<12}] "
                f"N0={self.N0:7.1f}  a2={self.a2:.4f}  "
                f"RMSE={m['RMSE']:.4f}  R={m['R']:.3f}  n={m['n']}")


# ════════════════════════════════════════════════════════════════════════════

class BaseCalibrator(ABC):
    """
    모든 교정 클래스의 추상 기반.

    Parameters
    ----------
    station_config : dict
        config/stations/{ID}.yaml 내용
    options : dict
        config/processing_options.yaml 내용
    """

    # 물리 상수 (Desilets et al.) — 하위 클래스에서 공유
    A0: float = 0.0808
    A1: float = 0.372

    def __init__(self, station_config: Dict, options: Dict):
        self.station_config = station_config
        self.options        = options
        self.station_id     = station_config["station_info"]["id"]

        soil = station_config.get("soil_properties", {})
        self.bulk_density = float(soil.get("bulk_density", 1.44))
        self.clay_content = float(soil.get("clay_content", 0.35))

        # lattice_water: YAML에 명시되면 사용, 없으면 None → calibrate 시 자동 계산
        lw = soil.get("lattice_water")
        self.lattice_water: Optional[float] = float(lw) if lw is not None else None

        self.result: Optional[CalibrationResult] = None

    # ── 하위 클래스가 반드시 구현 ──────────────────────────────────────────
    @abstractmethod
    def calibrate(self,
                  theta: np.ndarray,
                  N: np.ndarray,
                  **kwargs) -> CalibrationResult:
        """
        교정 수행.

        Parameters
        ----------
        theta : FDR 관측 토양수분 (m³/m³)
        N     : 보정된 중성자 계수율 (일평균)

        Returns
        -------
        CalibrationResult
        """

    # ── 공통 헬퍼 ──────────────────────────────────────────────────────────
    @staticmethod
    def vwc_from_N(N: np.ndarray, N0: float, a2: float) -> np.ndarray:
        """Desilets 역산: θ = a0 / (N/N0 − a1) − a2"""
        denom = N / N0 - BaseCalibrator.A1
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denom > 0,
                            BaseCalibrator.A0 / denom - a2,
                            np.nan)

    def _auto_lattice_water(self) -> float:
        """격자수 자동 계산 (crnpy 사용, 없으면 근사식)"""
        if self.lattice_water is not None:
            return self.lattice_water
        try:
            import crnpy
            lw = float(crnpy.lattice_water(clay_content=self.clay_content))
        except Exception:
            # 근사식: Wlat ≈ 0.02 + 0.5 * clay
            lw = 0.02 + 0.5 * self.clay_content * 0.1
        self.lattice_water = lw
        return lw

    def save(self, result_dir: Path) -> None:
        if self.result is None:
            raise RuntimeError("calibrate() 를 먼저 실행하세요.")
        self.result.save(result_dir)

    def summary(self) -> str:
        if self.result is None:
            return f"[{self.__class__.__name__}] Not calibrated yet."
        return self.result.summary()