# src/calibration/shp_joint.py
"""
SHPJointCalibrator  ──  N0 + a2 전역 동시 최적화
==================================================

[Standard vs SHP-Joint 비교]

  Standard:
    a2 = 0.115 (고정)
    for each candidate date:
      N0 = N_cal / (a0/(θ_cal + a2) + a1)
      RMSE = eval(전체 평가 기간)
    → 최소 RMSE 날짜 선택

  SHP-Joint:
    calibration period 전체 데이터를 한 번에 사용:
      minimize_{N0, a2}  std( VWC_raw(N, N0) - θ )
    → 단일 전역 해 (N0*, a2*)
    → 날짜 탐색 없음
    → 새 데이터가 쌓여도 calibration period가 고정되면 결과 불변

[수식 정리]
  a2는 mean offset:
    a2*(N0) = mean(VWC_raw(N, N0)) - mean(θ)
    where VWC_raw = a0 / (N/N0 - a1)  [a2=0 상태의 VWC]

  대입하면 N0만의 1D 최적화로 축약:
    minimize_{N0}  std( VWC_raw(N, N0) - θ )

  → std 최소 = bias 제거 후 잔차의 산포 최소
  → a2는 평균 편향(SHP 당량)을 흡수

[calibration period 역할]
  - candidate_start ~ candidate_end: 이 기간 데이터로 N0*, a2* 결정
  - 기간이 고정되면 결과도 고정 → 데이터 축적 후 재교정 불필요
  - analysis_start/end (공통)로 평가 기간은 별도 제어

[YAML 파라미터] calibration.shp_joint:
  candidate_start / candidate_end : 교정 기간 (전역 최적화에 사용)
  exclude_months                   : 제외 월 (null → 공통값)
  rmse_target                      : "theta_field" or "fdr_avg"
  a2_bounds                        : a2 탐색 범위
  n0_bounds                        : N0 유효 범위
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .base import BaseCalibrator, CalibrationResult


# ── Desilets (2010) 고정 상수 ────────────────────────────────────────────────
_A0: float = 0.0808
_A1: float = 0.372


class SHPJointCalibrator(BaseCalibrator):

    def __init__(self, station_cfg: Dict, options: Dict):
        super().__init__(station_cfg, options)

        cal = options.get("calibration", {})
        shp = cal.get("shp_joint", {})

        # 교정 기간
        self.cal_start, self.cal_end = self.get_cal_period("shp_joint")

        # 제외 월
        _global_excl = cal.get("exclude_months", [11, 12, 1, 2, 3])
        _local_excl  = shp.get("exclude_months", None)
        self.exclude_months: List[int] = (
            _local_excl if _local_excl is not None else _global_excl
        )

        self.rmse_target: str = shp.get("rmse_target", "theta_field")

        a2b = shp.get("a2_bounds", [0.0, 0.5])
        self.a2_bounds: Tuple[float, float] = (float(a2b[0]), float(a2b[1]))

        n0b = shp.get("n0_bounds", [300, 5000])
        self.n0_bounds: Tuple[float, float] = (float(n0b[0]), float(n0b[1]))

    # ─────────────────────────────────────────────────────────────────────────
    # 공개 인터페이스
    # ─────────────────────────────────────────────────────────────────────────

    def calibrate(self, matched_df: pd.DataFrame) -> CalibrationResult:
        print(f"\n{'='*60}")
        print(f"  SHPJointCalibrator  ─  {self.station_id}")
        print(f"  교정 기간  : {self.cal_start or '전체'} ~ {self.cal_end or '전체'}")
        print(f"  제외 월    : {self.exclude_months}")
        print(f"  RMSE 대상  : {self.rmse_target}")
        print(f"  a2 범위    : {self.a2_bounds}")
        print(f"{'='*60}")

        df = matched_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        if self.rmse_target not in df.columns:
            raise ValueError(
                f"rmse_target='{self.rmse_target}' 컬럼 없음. "
                f"사용 가능: {list(df.columns)}"
            )

        # 교정 기간: N0*, a2* 결정에 사용
        cal_df  = self._filter_candidates(df)
        # 평가 기간: VWC 계산 및 RMSE 평가
        eval_df = self._filter_eval(df)

        if len(cal_df) < 10:
            raise ValueError(f"교정 데이터 부족: {len(cal_df)}일")
        if len(eval_df) < 10:
            raise ValueError(f"평가 데이터 부족: {len(eval_df)}일")

        print(f"\n  교정 데이터: {len(cal_df)}일  "
              f"({cal_df['date'].min().date()} ~ {cal_df['date'].max().date()})")
        print(f"  평가 데이터: {len(eval_df)}일")

        # 교정 기간 데이터로 전역 최적화
        N_cal    = cal_df["N_corrected"].values.astype(float)
        theta_cal = cal_df[self.rmse_target].values.astype(float)

        N0_opt, a2_opt = self._global_optimize(N_cal, theta_cal)

        print(f"\n  ✅ 전역 최적화 완료")
        print(f"     N0*  = {N0_opt:.2f}")
        print(f"     a2*  = {a2_opt:.4f}")

        # 평가 기간 전체 VWC
        N_eval   = eval_df["N_corrected"].values.astype(float)
        obs_eval = eval_df[self.rmse_target].values.astype(float)
        vwc_eval = self._vwc(N_eval, N0_opt, a2_opt)

        valid = np.isfinite(vwc_eval) & np.isfinite(obs_eval)
        rmse  = float(np.sqrt(np.mean((vwc_eval[valid] - obs_eval[valid]) ** 2)))
        r     = float(np.corrcoef(vwc_eval[valid], obs_eval[valid])[0, 1])
        print(f"     RMSE = {rmse:.4f}  R = {r:.3f}  n = {valid.sum()}")

        result = CalibrationResult(
            method     = "shp_joint",
            station_id = self.station_id,
            N0         = N0_opt,
            a2         = a2_opt,
            vwc        = vwc_eval,
            obs        = obs_eval,
            cal_date   = None,   # 날짜 탐색 없음
            extra      = {
                "a0": _A0, "a1": _A1,
                "a2_opt":          a2_opt,
                "rmse_target":     self.rmse_target,
                "a2_bounds":       list(self.a2_bounds),
                "calibration_start": self.cal_start,
                "calibration_end":   self.cal_end,
                "exclude_months":  self.exclude_months,
                "n_cal_days":      len(cal_df),
                "n_eval_days":     len(eval_df),
            },
        )
        self.result = result
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 전역 최적화
    # ─────────────────────────────────────────────────────────────────────────

    def _global_optimize(self,
                         N:     np.ndarray,
                         theta: np.ndarray) -> Tuple[float, float]:
        """
        교정 기간 전체 데이터로 N0, a2 동시 최적화.

        핵심:
          a2*(N0) = mean(VWC_raw(N, N0)) - mean(theta)
          → N0에 대한 1D 최적화로 축약:
            minimize  std(VWC_raw(N, N0) - theta)
        """
        n0_lo, n0_hi = self.n0_bounds
        a2_lo, a2_hi = self.a2_bounds

        valid_mask = np.isfinite(N) & np.isfinite(theta)
        N_v, theta_v = N[valid_mask], theta[valid_mask]

        def objective(N0: float) -> float:
            denom = N_v / N0 - _A1
            vwc_raw = np.where(denom > 0, _A0 / denom, np.nan)
            ok = np.isfinite(vwc_raw)
            if ok.sum() < 5:
                return 1e9
            # a2 = mean offset → std가 최소화 기준
            return float(np.std(vwc_raw[ok] - theta_v[ok]))

        res = minimize_scalar(objective,
                              bounds=(n0_lo, n0_hi),
                              method="bounded")
        N0_opt = float(res.x)

        # a2* 결정
        denom   = N_v / N0_opt - _A1
        vwc_raw = np.where(denom > 0, _A0 / denom, np.nan)
        ok      = np.isfinite(vwc_raw)
        a2_opt  = float(np.mean(vwc_raw[ok] - theta_v[ok]))

        # a2 범위 클리핑
        a2_opt = float(np.clip(a2_opt, a2_lo, a2_hi))

        return N0_opt, a2_opt

    # ─────────────────────────────────────────────────────────────────────────
    # 필터
    # ─────────────────────────────────────────────────────────────────────────

    def _filter_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.filter_cal(df, self.cal_start, self.cal_end,
                               self.exclude_months, self.rmse_target)

    def _filter_eval(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.filter_eval(df, self.exclude_months, self.rmse_target)

    # ─────────────────────────────────────────────────────────────────────────
    # VWC
    # ─────────────────────────────────────────────────────────────────────────

    def _vwc(self, N: np.ndarray, N0: float, a2: float) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = N / N0 - _A1
            vwc   = np.where(denom > 0, _A0 / denom - a2, np.nan)
        return np.clip(np.asarray(vwc, dtype=float), 0.0, 1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # 결과 저장
    # ─────────────────────────────────────────────────────────────────────────

    def save_result(self, result_dir: Path) -> Path:
        if self.result is None:
            raise RuntimeError("calibrate() 를 먼저 실행하세요.")

        out_dir = result_dir / self.station_id / "shp_joint"
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = self.result.save(result_dir)
        print(f"\n  💾 결과 저장 → {out_dir}")
        print(f"     {json_path.name}")
        return out_dir

    # ─────────────────────────────────────────────────────────────────────────
    # 시각화
    # ─────────────────────────────────────────────────────────────────────────

    def plot_result(self, matched_df: pd.DataFrame, result_dir: Path) -> Path:
        if self.result is None:
            raise RuntimeError("calibrate() 를 먼저 실행하세요.")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        out_dir = result_dir / self.station_id / "shp_joint"
        out_dir.mkdir(parents=True, exist_ok=True)

        res = self.result
        m   = res.metrics
        ref_col = self.rmse_target

        df = matched_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df["VWC"]   = self._vwc(df["N_corrected"].values.astype(float),
                                res.N0, res.a2)
        eval_mask   = ~df["date"].dt.month.isin(self.exclude_months)

        fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                                 gridspec_kw={"height_ratios": [3, 1]},
                                 sharex=True)
        ax, ax_d = axes

        # 교정 기간 음영
        if self.cal_start and self.cal_end:
            ax.axvspan(pd.to_datetime(self.cal_start),
                       pd.to_datetime(self.cal_end),
                       color="lightyellow", alpha=0.5,
                       label=f"Cal. period ({self.cal_start} ~ {self.cal_end})", zorder=0)
        _shade_winter(ax, df["date"], self.exclude_months)

        ax.plot(df["date"], df[ref_col],
                color="black", lw=1.4, label=f"FDR ({ref_col})", zorder=5)
        ax.plot(df[eval_mask]["date"], df[eval_mask]["VWC"],
                color="tomato", lw=1.1, label="CRNP VWC (eval)", zorder=4)
        ax.plot(df[~eval_mask]["date"], df[~eval_mask]["VWC"],
                color="tomato", lw=0.7, ls="--", alpha=0.4,
                label="CRNP VWC (excluded)", zorder=4)

        txt = (f"N0   = {res.N0:.1f}\n"
               f"a2*  = {res.a2:.4f} (optimized)\n"
               f"a0={_A0}  a1={_A1}\n"
               f"─────────────\n"
               f"RMSE = {m['RMSE']:.4f}\n"
               f"R    = {m['R']:.3f}\n"
               f"NSE  = {m['NSE']:.3f}\n"
               f"n    = {m['n']}")
        ax.text(0.01, 0.97, txt, transform=ax.transAxes,
                va="top", ha="left", fontsize=9, fontfamily="DejaVu Sans",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          alpha=0.85, ec="gray"))

        ax.set_ylabel("VWC (m³/m³)", fontsize=11)
        ax.set_ylim(0, 0.6)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.set_title(
            f"[{self.station_id}] SHP-Joint Calibration  ─  FDR vs CRNP VWC\n"
            f"N0={res.N0:.1f}  |  a2*={res.a2:.4f}  |  "
            f"RMSE={m['RMSE']:.4f}  |  R={m['R']:.3f}",
            fontsize=11)
        ax.grid(axis="y", ls="--", alpha=0.4)

        diff = df["VWC"] - df[ref_col]
        ax_d.bar(df["date"], diff,
                 color=np.where(diff.fillna(0) >= 0, "tomato", "steelblue"),
                 width=1.5, alpha=0.7)
        ax_d.axhline(0, color="black", lw=0.8)
        ax_d.set_ylabel("Residual\n(VWC - th)", fontsize=9)
        ax_d.set_ylim(-0.3, 0.3)
        ax_d.grid(axis="y", ls="--", alpha=0.4)
        ax_d.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax_d.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax_d.xaxis.get_majorticklabels(), rotation=30, ha="right")
        plt.tight_layout()

        p1 = out_dir / f"{self.station_id}_shp_joint_calibration_timeseries.png"
        fig.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  📊 {p1.name}")

        # ── 산점도 (Scatter) ───────────────────────────────────────────
        from src.utils.plotting import plot_scatter
        p_scat = out_dir / f"{self.station_id}_shp_joint_calibration_scatter.png"
        plot_scatter(res.obs, res.vwc, "shp_joint", self.station_id, m, p_scat)

        return p1


# ─────────────────────────────────────────────────────────────────────────────

def _shade_winter(ax, dates: pd.Series, exclude_months) -> None:
    is_excl = dates.dt.month.isin(exclude_months)
    if not is_excl.any():
        return
    in_block, block_start = False, None
    for date, excl in zip(dates, is_excl):
        if excl and not in_block:
            block_start, in_block = date, True
        elif not excl and in_block:
            ax.axvspan(block_start, date, color="lightgray",
                       alpha=0.35, zorder=1)
            in_block = False
    if in_block:
        ax.axvspan(block_start, dates.iloc[-1], color="lightgray",
                   alpha=0.35, zorder=1)