# src/calibration/standard.py
"""
StandardCalibrator  ──  Desilets (2010) N0 교정
================================================

[논문: Desilets et al. (2010) Water Resour. Res., W11505]

식 (A1):
    θ(N) = a0 / (N/N0 - a1) - a2

고정 상수 (silica soil matrix, 논문 Table):
    a0 = 0.0808
    a1 = 0.372
    a2 = 0.115   (θ > 0.02 kg/kg)

N0 역산 (해석적):
    N0 = N / (a0/(θ + a2) + a1)

교정 절차:
    1. 후보 기간 내 매일 (θ_cal, N_cal) 로 N0_i 역산
    2. 이 N0_i 를 평가 기간 전체 N 에 적용 → VWC 시계열
    3. VWC vs 비교 대상(theta_field 또는 fdr_avg) RMSE 계산
    4. RMSE 최소 날짜 = 최적 교정 날짜, 해당 N0 채택

RMSE 비교 대상 (processing_options.yaml calibration.standard.rmse_target):
    "theta_field"  : Schron(2017) 거리가중 평균 [디폴트]
    "fdr_avg"      : reference_depths 지점 단순 평균 (원본 Desilets_01 방식)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseCalibrator, CalibrationResult


# ── Desilets (2010) 고정 상수 ────────────────────────────────────────────────
_A0: float = 0.0808
_A1: float = 0.372
_A2: float = 0.115   # 논문 Appendix A, for θ > 0.02 kg/kg


class StandardCalibrator(BaseCalibrator):

    def __init__(self, station_cfg: Dict, options: Dict):
        super().__init__(station_cfg, options)

        cal = options.get("calibration", {})
        std = cal.get("standard", {})

        # 교정 기간 (base.py get_cal_period 헬퍼 사용)
        self.cal_start, self.cal_end = self.get_cal_period("standard")

        # 제외 월
        _global_excl = cal.get("exclude_months", [11, 12, 1, 2, 3])
        _local_excl  = std.get("exclude_months", None)
        self.exclude_months: List[int] = (
            _local_excl if _local_excl is not None else _global_excl
        )

        self.rmse_target: str = std.get("rmse_target", "theta_field")

        bounds = std.get("n0_bounds", [300, 5000])
        self.n0_bounds = (float(bounds[0]), float(bounds[1]))

    # ─────────────────────────────────────────────────────────────────────────
    # 공개 인터페이스
    # ─────────────────────────────────────────────────────────────────────────

    def calibrate(self, matched_df: pd.DataFrame) -> CalibrationResult:
        print(f"\n{'='*60}")
        print(f"  StandardCalibrator  ─  {self.station_id}")
        print(f"  Desilets (2010): a0={_A0}  a1={_A1}  a2={_A2}  (고정)")
        print(f"  교정 기간  : {self.cal_start or '전체'} ~ {self.cal_end or '전체'}")
        print(f"  제외 월    : {self.exclude_months}")
        print(f"  RMSE 대상  : {self.rmse_target}")
        print(f"{'='*60}")

        df = matched_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # 비교 대상 컬럼 존재 확인
        if self.rmse_target not in df.columns:
            raise ValueError(
                f"rmse_target='{self.rmse_target}' 컬럼이 matched_df에 없음. "
                f"사용 가능: {[c for c in df.columns]}"
            )

        # 후보 날짜
        candidates = self._filter_candidates(df)
        if len(candidates) == 0:
            raise ValueError(
                "유효한 후보 날짜 없음. "
                "calibration_start/end 및 exclude_months 를 확인하세요."
            )

        # 평가 데이터
        eval_df = self._filter_eval(df)
        if len(eval_df) < 10:
            raise ValueError(f"평가 데이터 부족: {len(eval_df)}일")

        print(f"\n  후보 날짜  : {len(candidates)}개")
        print(f"  평가 데이터: {len(eval_df)}일")

        # 날짜 탐색
        best = self._search_best_date(candidates, eval_df)

        print(f"\n  ✅ 최적 교정 날짜: {best['date'].date()}")
        print(f"     N0   = {best['N0']:.2f}")
        print(f"     RMSE = {best['RMSE']:.4f}")
        print(f"     R    = {best['R']:.3f}")

        # 최적 N0로 평가 기간 전체 VWC
        N_eval   = eval_df["N_corrected"].values.astype(float)
        obs_eval = eval_df[self.rmse_target].values.astype(float)
        vwc_eval = self._vwc(N_eval, best["N0"])

        result = CalibrationResult(
            method     = "standard",
            station_id = self.station_id,
            N0         = best["N0"],
            a2         = _A2,
            vwc        = vwc_eval,
            obs        = obs_eval,
            cal_date   = best["date"].date(),
            extra      = {
                "a0": _A0, "a1": _A1, "a2": _A2,
                "rmse_target":     self.rmse_target,
                "calibration_start": self.cal_start,
                "calibration_end":   self.cal_end,
                "exclude_months":  self.exclude_months,
                "n_candidates":    len(candidates),
                "n_eval_days":     len(eval_df),
                "all_metrics":     best["all_metrics"],
            },
        )
        self.result = result
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 내부 헬퍼
    # ─────────────────────────────────────────────────────────────────────────

    def _filter_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """교정 기간 필터 → 파라미터 결정용 (base.filter_cal 사용)."""
        return self.filter_cal(df, self.cal_start, self.cal_end,
                               self.exclude_months, self.rmse_target)

    def _filter_eval(self, df: pd.DataFrame) -> pd.DataFrame:
        """평가 기간 필터 → VWC 계산용 (analysis 전체, base.filter_eval 사용)."""
        return self.filter_eval(df, self.exclude_months, self.rmse_target)

    def _search_best_date(self,
                          candidates: pd.DataFrame,
                          eval_df:    pd.DataFrame) -> Dict:
        N_eval   = eval_df["N_corrected"].values.astype(float)
        obs_eval = eval_df[self.rmse_target].values.astype(float)

        best_rmse   = np.inf
        best_N0     = None
        best_date   = None
        all_metrics: List[Dict] = []

        total = len(candidates)
        print(f"\n  날짜 탐색 중... ({total}개 후보)")

        for _, row in candidates.iterrows():
            theta_cal = float(row[self.rmse_target])
            N_cal     = float(row["N_corrected"])

            N0 = self._invert_N0(theta_cal, N_cal)
            if N0 is None:
                continue

            vwc  = self._vwc(N_eval, N0)
            valid = np.isfinite(vwc) & np.isfinite(obs_eval)
            if valid.sum() < 10:
                continue

            rmse = float(np.sqrt(np.mean((vwc[valid] - obs_eval[valid]) ** 2)))
            r    = float(np.corrcoef(vwc[valid], obs_eval[valid])[0, 1])

            all_metrics.append({
                "date": str(row["date"].date()),
                "N0":   round(N0,   3),
                "RMSE": round(rmse, 6),
                "R":    round(r,    6),
            })

            if rmse < best_rmse:
                best_rmse = rmse
                best_N0   = N0
                best_date = row["date"]

        if best_N0 is None:
            raise ValueError("유효한 교정 날짜를 찾지 못했습니다.")

        return {
            "date":        best_date,
            "N0":          best_N0,
            "RMSE":        best_rmse,
            "R":           next(m["R"] for m in all_metrics
                               if m["date"] == str(best_date.date())),
            "all_metrics": all_metrics,
        }

    def _invert_N0(self, theta: float, N: float) -> Optional[float]:
        """
        Desilets (2010) Eq. A1 해석적 역산.

        N/N0 = a0/(θ + a2) + a1
        N0   = N / (a0/(θ + a2) + a1)
        """
        try:
            denom = _A0 / (theta + _A2) + _A1
            if denom <= 0 or theta + _A2 <= 0:
                return None
            N0 = N / denom
            lo, hi = self.n0_bounds
            return float(N0) if lo <= N0 <= hi else None
        except Exception:
            return None

    def _vwc(self, N: np.ndarray, N0: float) -> np.ndarray:
        """
        Desilets (2010) Eq. A1:  θ = a0/(N/N0 - a1) - a2
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = N / N0 - _A1
            vwc   = np.where(denom > 0, _A0 / denom - _A2, np.nan)
        return np.clip(np.asarray(vwc, dtype=float), 0.0, 1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # 결과 저장
    # ─────────────────────────────────────────────────────────────────────────

    def save_result(self, result_dir: Path) -> Path:
        """
        교정 결과 저장.
          {result_dir}/{station_id}/standard/{station_id}_standard_calibration_result.json
          {result_dir}/{station_id}/standard/{station_id}_standard_all_metrics.xlsx
        """
        if self.result is None:
            raise RuntimeError("calibrate() 를 먼저 실행하세요.")

        out_dir = result_dir / self.station_id / "standard"
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = self.result.save(result_dir)

        metrics = self.result.extra.get("all_metrics", [])
        metrics_fname = f"{self.station_id}_standard_all_metrics.xlsx"
        if metrics:
            df_m = pd.DataFrame(metrics)
            df_m.to_excel(out_dir / metrics_fname, index=False)

        print(f"\n  💾 결과 저장 → {out_dir}")
        print(f"     {json_path.name}")
        if metrics:
            print(f"     {metrics_fname}  ({len(metrics)}개 후보)")

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

        out_dir = result_dir / self.station_id / "standard"
        out_dir.mkdir(parents=True, exist_ok=True)

        res = self.result
        m   = res.metrics

        df = matched_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df["VWC"]     = self._vwc(df["N_corrected"].values.astype(float), res.N0)
        eval_mask     = ~df["date"].dt.month.isin(self.exclude_months)
        ref_col       = self.rmse_target

        fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                                 gridspec_kw={"height_ratios": [3, 1]},
                                 sharex=True)
        ax, ax_d = axes

        if self.cal_start and self.cal_end:
            ax.axvspan(pd.to_datetime(self.cal_start),
                       pd.to_datetime(self.cal_end),
                       color="lightyellow", alpha=0.5,
                       label=f"Cal. period ({self.cal_start} ~ {self.cal_end})", zorder=0)

        # 제외 월 음영
        _shade_winter(ax, df["date"], self.exclude_months)

        ax.plot(df["date"], df[ref_col],
                color="black", lw=2.0,
                label=f"FDR ({ref_col})", zorder=5)
        ax.plot(df[eval_mask]["date"], df[eval_mask]["VWC"],
                color="tomato", lw=1.4, label="CRNP VWC (eval)", zorder=4)
        ax.plot(df[~eval_mask]["date"], df[~eval_mask]["VWC"],
                color="tomato", lw=0.8, ls="--", alpha=0.4,
                label="CRNP VWC (excluded)", zorder=4)

        # 교정 날짜 표시
        if res.cal_date:
            ax.axvline(pd.to_datetime(res.cal_date),
                       color="green", lw=2, ls=":", label=f"Cal. date ({res.cal_date})")

        txt = (f"N0   = {res.N0:.1f}\n"
               f"a0   = {_A0}  a1={_A1}  a2={_A2}\n"
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
            f"[{self.station_id}] Standard Calibration (Desilets 2010)  ─  "
            f"FDR vs CRNP VWC\n"
            f"N0={res.N0:.1f}  |  cal_date={res.cal_date}  |  "
            f"RMSE={m['RMSE']:.4f}  |  R={m['R']:.3f}",
            fontsize=11)
        ax.grid(axis="y", ls="--", alpha=0.4)

        diff = df["VWC"] - df[ref_col]
        ax_d.bar(df["date"], diff,
                 color=np.where(diff.fillna(0) >= 0, "tomato", "steelblue"),
                 width=1.5, alpha=0.7)
        ax_d.axhline(0, color="black", lw=0.8)
        ax_d.set_ylabel("Residual\n(VWC − θ)", fontsize=9)
        ax_d.set_ylim(-0.3, 0.3)
        ax_d.grid(axis="y", ls="--", alpha=0.4)
        ax_d.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax_d.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax_d.xaxis.get_majorticklabels(), rotation=30, ha="right")

        # ── 강수 막대 (보조 오른쪽 Y축, 역방향) ──────────────────────────
        from src.utils.plotting import add_rain_bars
        if "rain" in df.columns and df["rain"].notna().any():
            add_rain_bars(ax, df["date"], df["rain"])

        plt.tight_layout()

        p1 = out_dir / f"{self.station_id}_standard_calibration_timeseries.png"
        fig.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  📊 {p1.name}")

        # ── 산점도 ─────────────────────────────────────────────────────
        from src.utils.plotting import plot_scatter
        p_scat = out_dir / f"{self.station_id}_standard_calibration_scatter.png"
        plot_scatter(res.obs, res.vwc, "standard", self.station_id, m, p_scat)

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
            ax.axvspan(block_start, date, color="lightgray", alpha=0.35, zorder=1)
            in_block = False
    if in_block:
        ax.axvspan(block_start, dates.iloc[-1], color="lightgray",
                   alpha=0.35, zorder=1,
                   label=f"Excluded {exclude_months}")