# src/calibration/shp_2pt.py
"""
SHP2ptCalibrator  ──  슬라이딩 윈도우 기반 2점 해석적 풀이
===========================================================

[문제 배경]
단순 전체 기간 dry/wet 페어링은 계절에 따른 식생 체내 수분(DHP_veg) 변화를
포함하게 되어 a2 추정이 오염됨.

[해결 방법: 슬라이딩 윈도우]
  전체 후보 기간을 W일(window_days) 크기의 윈도우로 순회하며
  각 윈도우 내에서 dry/wet 페어를 추출 → a2 해석적 역산 반복
  → 모든 윈도우의 a2 분포에서 중앙값을 최종 SHP로 채택

  [이유] 짧은 윈도우 내에서는 DHP_veg 변화가 작아 식생 영향 최소화
         a2 분포의 안정성 자체가 SHP 추정 신뢰도 지표가 됨

[알고리즘]
  For each sliding window:
    1. 유효 데이터 추출 (theta_field, N_corrected 모두 유효)
    2. dry_percentile → θ_dry, N_dry
    3. wet_percentile → θ_wet, N_wet
    4. |θ_wet - θ_dry| < min_theta_diff 이면 스킵 (수분 차이 불충분)
    5. 2차 방정식 해석적 풀이 → a2_i, N0_i
    6. a2_i가 [a2_min, a2_max] 내이면 유효

  최종:
    a2* = median(유효 a2_i)
    N0  = N0 역산 (a2* 고정 후 전체 후보 기간 데이터로 재산출)

[수식 (Desilets)]
  A·a2² + B·a2 + C = 0
  A = a1 = 0.372
  B = a1·(θ1+θ2) + a0
  C = a1·θ1·θ2 + a0·(N1·θ1 - N2·θ2)/(N1-N2)
  a2* = (-B + √(B²-4AC)) / (2A)
  N0  = N1 / (a0/(θ1+a2*) + a1)

[YAML 파라미터] calibration.shp_2pt:
  candidate_start / candidate_end  : 후보 기간 (standard와 독립)
  window_days     : 슬라이딩 윈도우 크기 [일]
  step_days       : 윈도우 이동 간격 [일]
  dry_percentile  : 윈도우 내 θ_field 하위 N% → 건조점
  wet_percentile  : 윈도우 내 θ_field 상위 N% → 습윤점
  min_theta_diff  : dry/wet 최소 θ 차이 (이하면 스킵)
  a2_min / a2_max : a2 유효 범위
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseCalibrator, CalibrationResult


# ═══════════════════════════════════════════════════════════════════════════════
# 기본값
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULTS: Dict = {
    "window_days":     60,
    "step_days":        7,
    "dry_percentile":  10,
    "wet_percentile":  90,
    "min_theta_diff":  0.03,
    "a2_min":          0.0,   # SHP는 항상 양수
    "a2_max":          0.5,
}


def _cfg(options: Dict, key: str):
    return (options
            .get("calibration", {})
            .get("shp_2pt", {})
            .get(key, _DEFAULTS[key]))


# ═══════════════════════════════════════════════════════════════════════════════
# 해석적 풀이
# ═══════════════════════════════════════════════════════════════════════════════

def solve_2pt(theta1: float, N1: float,
              theta2: float, N2: float,
              a2_min: float = -0.1,
              a2_max: float =  0.5) -> Tuple[Optional[float], Optional[float]]:
    """
    2점 해석적 풀이.

    Parameters
    ----------
    theta1, N1 : 건조점 (θ1 < θ2)
    theta2, N2 : 습윤점

    Returns
    -------
    (a2, N0) or (None, None)
    """
    a0, a1 = 0.0808, 0.372

    if abs(N1 - N2) < 1e-6:
        return None, None

    A = a1
    B = a1 * (theta1 + theta2) + a0
    C = a1 * theta1 * theta2 + a0 * (N1 * theta1 - N2 * theta2) / (N1 - N2)

    disc = B ** 2 - 4 * A * C
    if disc < 0:
        return None, None

    sqrt_d = math.sqrt(disc)
    roots  = [(-B + sqrt_d) / (2 * A), (-B - sqrt_d) / (2 * A)]

    candidates = [r for r in roots if a2_min <= r <= a2_max]
    if not candidates:
        return None, None

    # 양수 근 우선 선택 (SHP는 물리적으로 0 이상)
    # C < 0 이면 근이 항상 (양수, 음수) 쌍 → 양수만 유효
    pos = [r for r in candidates if r >= 0]
    a2  = min(pos) if pos else min(candidates)
    denom = a0 / (theta1 + a2) + a1
    if denom <= 0:
        return None, None

    N0 = N1 / denom
    return (a2, float(N0)) if N0 > 0 else (None, None)


# ═══════════════════════════════════════════════════════════════════════════════
# SHP2ptCalibrator
# ═══════════════════════════════════════════════════════════════════════════════

class SHP2ptCalibrator(BaseCalibrator):

    def __init__(self, station_cfg: Dict, options: Dict):
        super().__init__(station_cfg, options)

        cal = options.get("calibration", {})
        shp = cal.get("shp_2pt", {})

        # exclude_months: shp_2pt 섹션 우선, 없으면 공통값
        _global_excl = cal.get("exclude_months", [11, 12, 1, 2, 3])
        _local_excl  = shp.get("exclude_months", None)
        self.exclude_months: List[int] = (
            _local_excl if _local_excl is not None else _global_excl
        )

        # 교정 기간
        self.cal_start, self.cal_end = self.get_cal_period("shp_2pt")

        # shp_2pt 전용 설정
        self.window_days:     int   = int(_cfg(options, "window_days"))
        self.step_days:       int   = int(_cfg(options, "step_days"))
        self.dry_pct:         int   = int(_cfg(options, "dry_percentile"))
        self.wet_pct:         int   = int(_cfg(options, "wet_percentile"))
        self.min_theta_diff:  float = float(_cfg(options, "min_theta_diff"))
        self.a2_min:          float = float(_cfg(options, "a2_min"))
        self.a2_max:          float = float(_cfg(options, "a2_max"))

    # ─────────────────────────────────────────────────────────────────────────
    # 공개 인터페이스
    # ─────────────────────────────────────────────────────────────────────────

    def calibrate(self, matched_df: pd.DataFrame) -> CalibrationResult:
        print(f"\n{'='*60}")
        print(f"  SHP2ptCalibrator  ─  {self.station_id}")
        print(f"  교정 기간: {self.cal_start or '전체'} ~ {self.cal_end or '전체'}")
        print(f"  슬라이딩 윈도우: {self.window_days}일  /  이동: {self.step_days}일")
        print(f"  dry={self.dry_pct}%  wet={self.wet_pct}%  "
              f"min_Δθ={self.min_theta_diff}")
        print(f"  a2 범위: [{self.a2_min}, {self.a2_max}]")
        print(f"{'='*60}")

        df = matched_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        cand = self._filter_candidates(df)
        if len(cand) < self.window_days // 2:
            raise ValueError(f"후보 데이터 부족: {len(cand)}일")

        # ── 슬라이딩 윈도우 탐색 ─────────────────────────────────────────
        window_results = self._sliding_window(cand)

        if len(window_results) == 0:
            raise ValueError(
                "유효한 윈도우 없음. window_days를 늘리거나 "
                "min_theta_diff를 줄여보세요."
            )

        a2_vals = np.array([r["a2"] for r in window_results])
        N0_vals = np.array([r["N0"] for r in window_results])

        a2_final = float(np.median(a2_vals))
        n_windows = len(window_results)

        print(f"\n  유효 윈도우: {n_windows}개")
        print(f"  a2 분포: median={a2_final:.4f}  "
              f"std={a2_vals.std():.4f}  "
              f"[{a2_vals.min():.4f}, {a2_vals.max():.4f}]")

        # ── a2 고정 후 N0 재역산 (전체 후보 기간 가중 평균) ──────────────
        N0_final = self._refit_N0(cand, a2_final)
        print(f"  N0 재역산: {N0_final:.2f}  (a2={a2_final:.4f} 고정)")

        # ── 평가 기간 VWC ─────────────────────────────────────────────────
        eval_df  = self._filter_eval(df)
        N_eval   = eval_df["N_corrected"].values.astype(float)
        obs_eval = eval_df["theta_field"].values.astype(float)
        vwc_eval = self._vwc(N_eval, N0_final, a2_final)

        m = _quick_metrics(obs_eval, vwc_eval)
        print(f"  평가: RMSE={m['RMSE']:.4f}  R={m['R']:.3f}  "
              f"n={m['n']}  Bias={m['Bias']:+.4f}")

        result = CalibrationResult(
            method     = "shp_2pt",
            station_id = self.station_id,
            N0         = N0_final,
            a2         = a2_final,
            vwc        = vwc_eval,
            obs        = obs_eval,
            cal_date   = None,
            extra      = {
                "a2_median":       a2_final,
                "a2_std":          float(a2_vals.std()),
                "a2_all":          a2_vals.tolist(),
                "N0_all":          N0_vals.tolist(),
                "n_windows":       n_windows,
                "window_days":     self.window_days,
                "step_days":       self.step_days,
                "calibration_start": self.cal_start,
                "calibration_end":   self.cal_end,
                "exclude_months":  self.exclude_months,
                "n_eval_days":     len(eval_df),
                "window_results":  [
                    {k: (str(v) if isinstance(v, pd.Timestamp) else v)
                     for k, v in r.items()}
                    for r in window_results
                ],
            },
        )
        self.result = result
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 슬라이딩 윈도우
    # ─────────────────────────────────────────────────────────────────────────

    def _sliding_window(self, df: pd.DataFrame) -> List[Dict]:
        """
        슬라이딩 윈도우로 a2, N0 반복 역산.
        반환: 유효한 윈도우 결과 리스트
        """
        results: List[Dict] = []
        dates   = df["date"].values
        t_start = df["date"].min()
        t_end   = df["date"].max() - pd.Timedelta(days=self.window_days)

        win  = pd.Timedelta(days=self.window_days)
        step = pd.Timedelta(days=self.step_days)
        cur  = t_start

        while cur <= t_end:
            win_end = cur + win
            mask    = (df["date"] >= cur) & (df["date"] < win_end)
            sub     = df[mask].dropna(subset=["theta_field", "N_corrected"])

            cur += step

            if len(sub) < 10:
                continue

            theta = sub["theta_field"].values
            N     = sub["N_corrected"].values

            # 백분위수 대표점
            theta_dry = np.percentile(theta, self.dry_pct)
            theta_wet = np.percentile(theta, self.wet_pct)

            if (theta_wet - theta_dry) < self.min_theta_diff:
                continue

            # 대표점: 백분위수에 가장 가까운 날의 값
            idx_dry = np.argmin(np.abs(theta - theta_dry))
            idx_wet = np.argmin(np.abs(theta - theta_wet))

            t1, N1 = float(theta[idx_dry]), float(N[idx_dry])
            t2, N2 = float(theta[idx_wet]), float(N[idx_wet])

            a2, N0 = solve_2pt(t1, N1, t2, N2, self.a2_min, self.a2_max)
            if a2 is None:
                continue

            results.append({
                "window_start": str(cur.date()),
                "window_end":   str(win_end.date()),
                "theta_dry":    round(t1, 5),
                "N_dry":        round(N1, 2),
                "theta_wet":    round(t2, 5),
                "N_wet":        round(N2, 2),
                "a2":           round(a2, 6),
                "N0":           round(N0, 3),
            })

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # a2 고정 후 N0 재역산
    # ─────────────────────────────────────────────────────────────────────────

    def _refit_N0(self, df: pd.DataFrame, a2: float) -> float:
        """
        a2 고정 후 후보 기간 전체 데이터로 N0 재역산.

        각 행에서 N0_i = N / (a0/(θ+a2) + a1) 를 계산하고
        중앙값을 최종 N0로 사용.
        """
        a0, a1 = 0.0808, 0.372
        theta  = df["theta_field"].values.astype(float)
        N      = df["N_corrected"].values.astype(float)

        denom  = a0 / (theta + a2) + a1
        valid  = (denom > 0) & np.isfinite(denom) & np.isfinite(N)
        N0_arr = N[valid] / denom[valid]

        # 물리 범위 필터
        N0_arr = N0_arr[(N0_arr > 300) & (N0_arr < 5000)]

        if len(N0_arr) == 0:
            raise ValueError("N0 재역산 실패: 유효한 데이터 없음")

        return float(np.median(N0_arr))

    # ─────────────────────────────────────────────────────────────────────────
    # 필터
    # ─────────────────────────────────────────────────────────────────────────

    def _filter_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = pd.Series(True, index=df.index)
        if self.cal_start:
            mask &= df["date"] >= pd.to_datetime(self.cal_start)
        if self.cal_end:
            mask &= df["date"] <= pd.to_datetime(self.cal_end)
        mask &= ~df["date"].dt.month.isin(self.exclude_months)
        mask &= df["theta_field"].notna() & df["N_corrected"].notna()
        return df[mask].copy()

    def _filter_eval(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = (
            ~df["date"].dt.month.isin(self.exclude_months)
            & df["theta_field"].notna()
            & df["N_corrected"].notna()
        )
        return df[mask].copy()

    # ─────────────────────────────────────────────────────────────────────────
    # VWC
    # ─────────────────────────────────────────────────────────────────────────

    def _vwc(self, N: np.ndarray, N0: float, a2: float) -> np.ndarray:
        a0, a1 = self.A0, self.A1
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = N / N0 - a1
            vwc   = np.where(denom > 0, a0 / denom - a2, np.nan)
        return np.clip(np.asarray(vwc, dtype=float), 0.0, 1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # 시각화
    # ─────────────────────────────────────────────────────────────────────────

    def plot_result(self, matched_df: pd.DataFrame, result_dir: Path) -> Path:
        """
        [1] calibration_timeseries.png  — FDR vs CRNP VWC
        [2] a2_distribution.png         — 슬라이딩 윈도우별 a2 분포 (안정성 진단)
        [3] shp_curve.png               — Desilets 곡선 비교 (a2=0.115 기본값 vs SHP-2pt 역산값)
        """
        if self.result is None:
            raise RuntimeError("calibrate() 를 먼저 실행하세요.")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        res     = self.result
        m       = res.metrics
        out_dir = result_dir / self.station_id / "shp_2pt"
        out_dir.mkdir(parents=True, exist_ok=True)

        df = matched_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df["VWC"]   = self._vwc(df["N_corrected"].values.astype(float),
                                res.N0, res.a2)
        eval_mask   = ~df["date"].dt.month.isin(self.exclude_months)

        # ══ Plot 1: 시계열 ════════════════════════════════════════════════
        fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                                 gridspec_kw={"height_ratios": [3, 1]},
                                 sharex=True)
        ax, ax_d = axes

        if self.cal_start and self.cal_end:
            ax.axvspan(pd.to_datetime(self.cal_start),
                       pd.to_datetime(self.cal_end),
                       color="lightyellow", alpha=0.5,
                       label=f"Cal. period ({self.cal_start} ~ {self.cal_end})", zorder=0)
        _shade_winter(ax, df["date"], self.exclude_months)

        ax.plot(df["date"], df["theta_field"],
                color="black", lw=1.4,
                label=r"FDR $\theta_{field}$", zorder=3)
        ax.plot(df[eval_mask]["date"], df[eval_mask]["VWC"],
                color="tomato", lw=1.1, label="CRNP VWC (eval)", zorder=4)
        ax.plot(df[~eval_mask]["date"], df[~eval_mask]["VWC"],
                color="tomato", lw=0.7, ls="--", alpha=0.4,
                label="CRNP VWC (excluded)", zorder=4)

        # 각 슬라이딩 윈도우의 dry/wet 날짜를 연한 수직선으로 표시
        # (모든 윈도우를 표시하면 복잡하므로 dry는 갈색, wet은 파랑 반투명)
        wr = res.extra.get("window_results", [])
        for r in wr:
            if r.get("theta_dry") is not None and r.get("theta_wet") is not None:
                ax.axvline(pd.to_datetime(r["window_start"]),
                           color="saddlebrown", lw=0.5, ls=":", alpha=0.3, zorder=2)
                ax.axvline(pd.to_datetime(r["window_end"]),
                           color="navy", lw=0.5, ls=":", alpha=0.3, zorder=2)
        # 범례용 더미 라인 (첫 번째만)
        if wr:
            ax.plot([], [], color="saddlebrown",
                       lw=0.8, ls=":", alpha=0.5, label="Window start (dry)")
            ax.plot([], [], color="navy",
                       lw=0.8, ls=":", alpha=0.5, label="Window end (wet)")

        a2_std = res.extra.get("a2_std", 0)
        n_wins = res.extra.get("n_windows", "?")
        txt = (f"a2   = {res.a2:.4f} +/- {a2_std:.4f}\n"
               f"N0   = {res.N0:.1f}\n"
               f"wins = {n_wins}\n"
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
            f"[{self.station_id}] SHP 2-Point (Sliding Window)  —  "
            f"FDR vs CRNP VWC\n"
            f"a2={res.a2:.4f}±{a2_std:.4f}  N0={res.N0:.1f}  "
            f"RMSE={m['RMSE']:.4f}  R={m['R']:.3f}",
            fontsize=11)
        ax.grid(axis="y", ls="--", alpha=0.4)

        diff = df["VWC"] - df["theta_field"]
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
        plt.tight_layout()
        p1 = out_dir / "calibration_timeseries.png"
        fig.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  📊 [1] {p1.name}")

        # ══ Plot 2: a2 분포 (슬라이딩 윈도우 안정성) ═════════════════════
        wr = res.extra.get("window_results", [])
        if wr:
            fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

            # 왼쪽: 시간에 따른 a2 추이
            ax_ts = axes2[0]
            starts = [pd.to_datetime(r["window_start"]) for r in wr]
            a2s    = [r["a2"] for r in wr]
            ax_ts.scatter(starts, a2s, color="steelblue", s=20, alpha=0.6)
            ax_ts.axhline(res.a2, color="tomato", lw=2,
                          label=f"median a2 = {res.a2:.4f}")
            ax_ts.axhspan(res.a2 - a2_std, res.a2 + a2_std,
                          color="tomato", alpha=0.15, label="±1 std")
            ax_ts.set_xlabel("Window start date")
            ax_ts.set_ylabel("a2 (SHP equivalent)")
            ax_ts.set_title("a2 over time (sliding windows)")
            ax_ts.legend(fontsize=9)
            ax_ts.grid(ls="--", alpha=0.4)
            ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=30, ha="right")

            # 오른쪽: a2 히스토그램
            ax_hist = axes2[1]
            ax_hist.hist(a2s, bins=30, color="steelblue", alpha=0.7,
                         edgecolor="white")
            ax_hist.axvline(res.a2, color="tomato", lw=2,
                            label=f"median = {res.a2:.4f}")
            ax_hist.set_xlabel("a2 (SHP equivalent)")
            ax_hist.set_ylabel("Count")
            ax_hist.set_title(f"a2 Distribution  (n={len(a2s)} windows)")
            ax_hist.legend(fontsize=9)
            ax_hist.grid(ls="--", alpha=0.4)

            plt.suptitle(f"[{self.station_id}] SHP Stability Diagnosis",
                         fontsize=12)
            plt.tight_layout()
            p2 = out_dir / "a2_distribution.png"
            fig2.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(fig2)
            print(f"  📊 [2] {p2.name}")

        # ══ Plot 3: Desilets 곡선 비교 ════════════════════════════════════
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        theta_r = np.linspace(0.01, 0.6, 300)
        a0, a1  = 0.0808, 0.372

        for a2_v, lbl, col, ls in [
            (0.115,  "Desilets default (a2=0.115)", "orange", "--"),
            (res.a2, f"SHP-2pt (a2={res.a2:.4f})",  "tomato", "-"),
        ]:
            N_c = res.N0 * (a0 / (theta_r + a2_v) + a1)
            ax3.plot(theta_r, N_c, color=col, ls=ls, lw=2.5, label=lbl)

        valid = df[df["N_corrected"].notna() & df["theta_field"].notna()]
        ax3.scatter(valid["theta_field"], valid["N_corrected"],
                    color="steelblue", alpha=0.15, s=10, label="Observations")
        ax3.set_xlabel(r"$\theta_{field}$ (m³/m³)", fontsize=11)
        ax3.set_ylabel("N_corrected (counts)", fontsize=11)
        ax3.set_title(
            f"[{self.station_id}] Desilets Calibration Curve\n"
            f"N0={res.N0:.1f}  |  "
            f"a2: 0.115 (default) → {res.a2:.4f} (SHP-2pt)",
            fontsize=11,
        )
        ax3.legend(fontsize=9); ax3.grid(ls="--", alpha=0.4)
        ax3.set_xlim(0, 0.65)
        plt.tight_layout()
        p3 = out_dir / "shp_curve.png"
        fig3.savefig(p3, dpi=150, bbox_inches="tight"); plt.close(fig3)
        print(f"  📊 [3] {p3.name}")

        return p1

    # ─────────────────────────────────────────────────────────────────────────
    # 결과 저장
    # ─────────────────────────────────────────────────────────────────────────

    def save_result(self, result_dir: Path) -> Path:
        if self.result is None:
            raise RuntimeError("calibrate() 를 먼저 실행하세요.")
        out_dir = result_dir / self.station_id / "shp_2pt"
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.result.save(result_dir)
        print(f"\n  💾 결과 저장 → {out_dir}")
        print(f"     {json_path.name}")
        return out_dir


# ═══════════════════════════════════════════════════════════════════════════════
# 헬퍼
# ═══════════════════════════════════════════════════════════════════════════════

def _quick_metrics(obs: np.ndarray, vwc: np.ndarray) -> Dict:
    valid = np.isfinite(obs) & np.isfinite(vwc)
    if valid.sum() < 2:
        return {"RMSE": np.nan, "R": np.nan, "Bias": np.nan, "n": 0}
    o, v  = obs[valid], vwc[valid]
    rmse  = float(np.sqrt(np.mean((v - o) ** 2)))
    bias  = float(np.mean(v - o))
    r     = float(np.corrcoef(o, v)[0, 1])
    ss_r  = np.sum((o - v) ** 2)
    ss_t  = np.sum((o - o.mean()) ** 2)
    nse   = float(1 - ss_r / ss_t) if ss_t > 0 else np.nan
    return {"RMSE": rmse, "R": r, "Bias": bias, "NSE": nse,
            "n": int(valid.sum())}


def _shade_winter(ax, dates: pd.Series, exclude_months: List[int]) -> None:
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