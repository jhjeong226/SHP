# src/calibration/standard.py
"""
StandardCalibrator (Desilet 방식)
==================================
날짜 탐색(Date Search)으로 N0를 결정합니다.

알고리즘 (Desilet_01_Calibration_HC.py 기반):
  1. 전체 기간의 일별 매칭 데이터(matched_df)에서
     성장기(4~10월) 후보 날짜를 추출
  2. 각 후보 날짜의 단일 (θ, N) 쌍으로 N0 결정
     minimize (VWC(N_cal, N0) - θ_cal)²
  3. 이 N0를 전체 평가 기간(4~10월)에 적용
     VWC 계산 → FDR_avg(10cm)와 RMSE 비교
  4. RMSE 최소 날짜 = 최적 교정 날짜, 해당 N0 채택
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .base import BaseCalibrator, CalibrationResult


class StandardCalibrator(BaseCalibrator):

    def __init__(self, station_config: Dict, options: Dict):
        super().__init__(station_config, options)
        cal_opts = options.get("calibration", {})
        self.growing_months = list(range(4, 11))   # 4~10월

    # ── 핵심 메서드 ──────────────────────────────────────────────────────────

    def calibrate(self,
                  matched_df: pd.DataFrame,
                  eval_df:    pd.DataFrame,
                  **kwargs) -> CalibrationResult:
        """
        Parameters
        ----------
        matched_df : date | theta | N   (전체 기간 일별 매칭)
        eval_df    : date | FDR_avg     (10cm 일평균)
        """
        wlat = self._auto_lattice_water()

        matched = matched_df.copy()
        matched["date"] = pd.to_datetime(matched["date"])

        edf = eval_df.copy()
        edf["date"] = pd.to_datetime(edf["date"])

        # ── 평가 DataFrame: matched + FDR_avg 병합, 성장기만 ─────────────────
        eval_merged = pd.merge(
            matched.rename(columns={"date": "date"}),
            edf.rename(columns={"date": "date"}),
            on="date", how="inner"
        )
        eval_growing = eval_merged[
            eval_merged["date"].dt.month.isin(self.growing_months)
        ].copy()

        if len(eval_growing) < 10:
            raise ValueError(
                f"평가 데이터 부족: {len(eval_growing)}일 "
                f"(성장기 4~10월, FDR_avg 필요)"
            )

        # ── 후보 날짜: 성장기 matched 데이터 ─────────────────────────────────
        candidates = matched[
            matched["date"].dt.month.isin(self.growing_months) &
            matched["theta"].notna() &
            matched["N"].notna()
        ]

        print(f"\n[StandardCalibrator] {self.station_id}")
        print(f"  Wlat={wlat:.4f}  bulk_density={self.bulk_density}")
        print(f"  후보 날짜: {len(candidates)}개  평가 데이터: {len(eval_growing)}일")

        # ── 날짜 탐색 ─────────────────────────────────────────────────────────
        best_rmse  = float("inf")
        best_N0    = None
        best_date  = None
        all_metrics = []

        N_eval  = eval_growing["N"].values.astype(float)
        obs_eval = eval_growing["FDR_avg"].values.astype(float)

        for _, row in candidates.iterrows():
            theta_cal = float(row["theta"])
            N_cal     = float(row["N"])

            N0 = self._single_point_N0(theta_cal, N_cal, wlat)
            if N0 is None:
                continue

            vwc = self._predict_vwc(N_eval, N0, wlat)
            valid = np.isfinite(vwc) & np.isfinite(obs_eval)
            if valid.sum() < 10:
                continue

            rmse = float(np.sqrt(np.mean((vwc[valid] - obs_eval[valid]) ** 2)))
            r    = float(np.corrcoef(vwc[valid], obs_eval[valid])[0, 1])

            all_metrics.append({"date": row["date"], "N0": N0,
                                 "RMSE": rmse, "R": r})

            if rmse < best_rmse:
                best_rmse = rmse
                best_N0   = N0
                best_date = row["date"]

        if best_N0 is None:
            raise ValueError("유효한 교정 날짜를 찾지 못했습니다.")

        print(f"  ✅ 최적 날짜: {best_date.date()}  "
              f"N0={best_N0:.1f}  RMSE={best_rmse:.4f}")

        # ── 최적 N0로 최종 VWC ────────────────────────────────────────────────
        vwc_final = self._predict_vwc(N_eval, best_N0, wlat)

        self.result = CalibrationResult(
            method     = "standard",
            station_id = self.station_id,
            N0         = best_N0,
            a2         = wlat,
            vwc        = vwc_final,
            obs        = obs_eval,
            cal_date   = best_date.date(),
            extra      = {
                "Wlat":         wlat,
                "bulk_density": self.bulk_density,
                "n_candidates": len(candidates),
                "n_eval_days":  int(valid.sum()),
                "all_metrics":  [
                    {k: (str(v) if hasattr(v, 'date') else float(v) if isinstance(v, float) else v)
                     for k, v in m.items()}
                    for m in all_metrics
                ],
            },
        )
        return self.result

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────────

    def _single_point_N0(self, theta: float, N: float,
                          wlat: float) -> Optional[float]:
        """단일 (θ, N) 쌍으로 N0 결정."""
        try:
            def obj(N0):
                vwc = self._predict_vwc(np.array([N]), N0, wlat)
                return (float(vwc[0]) - theta) ** 2

            res = minimize_scalar(obj, bounds=(300, 5000), method="bounded")
            return float(res.x) if res.success else None
        except Exception:
            return None