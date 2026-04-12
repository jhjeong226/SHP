# src/calibration/shp_2pt.py
"""
SHP2ptCalibrator  ──  2점 해석적 풀이로 N0 + a2 동시 역산
===========================================================

핵심 아이디어 (SHP and DHP.md):
  기존 Standard 방식은 a2=0.115 고정 → N0가 SHP 오차를 흡수 → Lumping Problem
  본 방법은 건조·습윤 2점의 (θ, N) 쌍으로 연립방정식을 세워
  N0와 a2를 수치 최적화 없이 2차 방정식 해석적 해로 동시에 역산한다.

수식 (Desilets 방정식):
  N/N0 = a0/(θ + a2) + a1    (a0=0.0808, a1=0.372 고정)

2점 연립방정식에서 N0 소거 → a2에 대한 2차 방정식:
  A·a2² + B·a2 + C = 0
  A = a1
  B = a1·(θ1 + θ2) + a0
  C = a1·θ1·θ2 + a0·(N1·θ1 − N2·θ2) / (N1 − N2)

근의 공식:
  a2* = (−B + √(B²−4AC)) / (2A)   [물리적으로 유효한 양의 근]

N0 역산:
  N0 = N1 / (a0/(θ1 + a2*) + a1)

대표점 선택 (YAML calibration.shp_2pt):
  dry_percentile : 후보 기간 θ_field 하위 N% 날짜 → 건조 대표점
  wet_percentile : 후보 기간 θ_field 상위 N% 날짜 → 습윤 대표점
  window_days    : 해당 날짜 ±window_days 이내 평균으로 대표점 산출
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
    "dry_percentile": 10,
    "wet_percentile": 90,
    "window_days":     5,
    "a2_min":        -0.1,
    "a2_max":         0.5,
}


def _cfg(options: Dict, key: str):
    return (options
            .get("calibration", {})
            .get("shp_2pt", {})
            .get(key, _DEFAULTS[key]))


# ═══════════════════════════════════════════════════════════════════════════════
# 해석적 풀이 핵심 함수
# ═══════════════════════════════════════════════════════════════════════════════

def solve_2pt(theta1: float, N1: float,
              theta2: float, N2: float,
              a2_min: float = -0.1,
              a2_max: float =  0.5) -> Tuple[Optional[float], Optional[float]]:
    """
    2점 해석적 풀이.

    Parameters
    ----------
    theta1, N1 : 건조 대표점 (θ_dry, N_dry)  — θ1 < θ2 이어야 함
    theta2, N2 : 습윤 대표점 (θ_wet, N_wet)

    Returns
    -------
    (a2, N0) or (None, None) if no physical solution
    """
    a0, a1 = 0.0808, 0.372

    if abs(N1 - N2) < 1e-6:
        return None, None   # N이 같으면 분모 0

    # 2차 방정식 계수
    A = a1
    B = a1 * (theta1 + theta2) + a0
    C = a1 * theta1 * theta2 + a0 * (N1 * theta1 - N2 * theta2) / (N1 - N2)

    discriminant = B**2 - 4 * A * C
    if discriminant < 0:
        return None, None   # 실수해 없음

    # 두 근 중 물리적으로 유효한 것(a2 > a2_min) 선택
    sqrt_d = math.sqrt(discriminant)
    roots = [(-B + sqrt_d) / (2 * A), (-B - sqrt_d) / (2 * A)]

    a2_candidates = [r for r in roots if a2_min <= r <= a2_max]
    if not a2_candidates:
        return None, None

    # 여러 개면 a2가 더 작은(보수적) 값 선택
    a2 = min(a2_candidates)

    # N0 역산
    denom = a0 / (theta1 + a2) + a1
    if denom <= 0:
        return None, None
    N0 = N1 / denom

    if N0 <= 0:
        return None, None

    return a2, N0


# ═══════════════════════════════════════════════════════════════════════════════
# SHP2ptCalibrator
# ═══════════════════════════════════════════════════════════════════════════════

class SHP2ptCalibrator(BaseCalibrator):
    """
    Parameters
    ----------
    station_cfg : dict   (YAML station 섹션)
    options     : dict   (YAML processing_options 섹션)
    """

    def __init__(self, station_cfg: Dict, options: Dict):
        super().__init__(station_cfg, options)

        cal = options.get("calibration", {})
        self.candidate_start: str       = cal.get("candidate_start", "")
        self.candidate_end:   str       = cal.get("candidate_end",   "")
        self.exclude_months:  List[int] = cal.get("exclude_months",  [11, 12, 1, 2])

        self.dry_pct:    int   = int(_cfg(options, "dry_percentile"))
        self.wet_pct:    int   = int(_cfg(options, "wet_percentile"))
        self.window:     int   = int(_cfg(options, "window_days"))
        self.a2_min:   float   = float(_cfg(options, "a2_min"))
        self.a2_max:   float   = float(_cfg(options, "a2_max"))

    # ─────────────────────────────────────────────────────────────────────────
    # 공개 인터페이스
    # ─────────────────────────────────────────────────────────────────────────

    def calibrate(self, matched_df: pd.DataFrame) -> CalibrationResult:
        """
        Parameters
        ----------
        matched_df : DataMatcher.run() 출력  (date | theta_field | N_corrected)

        Returns
        -------
        CalibrationResult  (method="shp_2pt")
        """
        print(f"\n{'='*60}")
        print(f"  SHP2ptCalibrator  ─  {self.station_id}")
        print(f"  dry_pct={self.dry_pct}%  wet_pct={self.wet_pct}%  "
              f"window=±{self.window}일")
        print(f"  후보 기간: {self.candidate_start} ~ {self.candidate_end}")
        print(f"  a2 유효 범위: [{self.a2_min}, {self.a2_max}]")
        print(f"{'='*60}")

        df = matched_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # ── 후보 기간 필터링 ──────────────────────────────────────────────
        cand = self._filter_candidates(df)
        if len(cand) < 20:
            raise ValueError(f"후보 데이터 부족: {len(cand)}일 (최소 20일 필요)")

        # ── 건조 / 습윤 대표점 선택 ──────────────────────────────────────
        dry_pt, wet_pt = self._select_reference_points(cand, df)

        print(f"\n  건조 대표점: θ={dry_pt['theta']:.4f}  "
              f"N={dry_pt['N']:.1f}  ({dry_pt['date']})")
        print(f"  습윤 대표점: θ={wet_pt['theta']:.4f}  "
              f"N={wet_pt['N']:.1f}  ({wet_pt['date']})")

        # ── 해석적 풀이 ───────────────────────────────────────────────────
        a2, N0 = solve_2pt(
            dry_pt["theta"], dry_pt["N"],
            wet_pt["theta"], wet_pt["N"],
            a2_min=self.a2_min,
            a2_max=self.a2_max,
        )

        if a2 is None:
            raise ValueError(
                "유효한 해석적 해 없음. "
                "건조·습윤 대표점 차이가 충분하지 않거나 물리 범위를 벗어났습니다. "
                f"dry=({dry_pt['theta']:.4f}, {dry_pt['N']:.1f})  "
                f"wet=({wet_pt['theta']:.4f}, {wet_pt['N']:.1f})"
            )

        print(f"\n  ✅ 해석적 해: a2={a2:.4f}  N0={N0:.2f}")

        # ── 평가 기간 전체 VWC 계산 ──────────────────────────────────────
        eval_df  = self._filter_eval(df)
        N_eval   = eval_df["N_corrected"].values.astype(float)
        obs_eval = eval_df["theta_field"].values.astype(float)
        vwc_eval = self._vwc(N_eval, N0, a2)

        m = _quick_metrics(obs_eval, vwc_eval)
        print(f"  평가: RMSE={m['RMSE']:.4f}  R={m['R']:.3f}  "
              f"n={m['n']}  Bias={m['Bias']:+.4f}")

        result = CalibrationResult(
            method     = "shp_2pt",
            station_id = self.station_id,
            N0         = N0,
            a2         = a2,
            vwc        = vwc_eval,
            obs        = obs_eval,
            cal_date   = None,
            extra      = {
                "dry_point":      dry_pt,
                "wet_point":      wet_pt,
                "dry_percentile": self.dry_pct,
                "wet_percentile": self.wet_pct,
                "window_days":    self.window,
                "candidate_start": self.candidate_start,
                "candidate_end":   self.candidate_end,
                "exclude_months":  self.exclude_months,
                "n_candidates":    len(cand),
                "n_eval_days":     len(eval_df),
            },
        )
        self.result = result
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 대표점 선택
    # ─────────────────────────────────────────────────────────────────────────

    def _filter_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = pd.Series(True, index=df.index)
        if self.candidate_start:
            mask &= df["date"] >= pd.to_datetime(self.candidate_start)
        if self.candidate_end:
            mask &= df["date"] <= pd.to_datetime(self.candidate_end)
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

    def _select_reference_points(
            self,
            cand: pd.DataFrame,
            full_df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """
        백분위수 기반 건조·습윤 날짜 탐색 → ±window_days 이내 평균으로 대표점 산출.

        Returns
        -------
        (dry_dict, wet_dict)  각각 {"date", "theta", "N"} 포함
        """
        theta = cand["theta_field"]
        dry_threshold = np.percentile(theta, self.dry_pct)
        wet_threshold = np.percentile(theta, self.wet_pct)

        # 건조: theta가 dry_threshold에 가장 가까운 날
        dry_idx  = (theta - dry_threshold).abs().idxmin()
        dry_date = cand.loc[dry_idx, "date"]

        # 습윤: theta가 wet_threshold에 가장 가까운 날
        wet_idx  = (theta - wet_threshold).abs().idxmin()
        wet_date = cand.loc[wet_idx, "date"]

        dry_pt = self._window_average(full_df, dry_date, label="dry")
        wet_pt = self._window_average(full_df, wet_date, label="wet")

        # 건조 < 습윤 조건 확인
        if dry_pt["theta"] >= wet_pt["theta"]:
            raise ValueError(
                f"건조 θ({dry_pt['theta']:.4f}) ≥ 습윤 θ({wet_pt['theta']:.4f}). "
                "percentile 설정을 확인하세요."
            )

        return dry_pt, wet_pt

    def _window_average(self,
                        df: pd.DataFrame,
                        center_date: pd.Timestamp,
                        label: str) -> Dict:
        """center_date ± window_days 이내 유효 데이터의 평균."""
        window = pd.Timedelta(days=self.window)
        mask   = (
            (df["date"] >= center_date - window) &
            (df["date"] <= center_date + window) &
            df["theta_field"].notna() &
            df["N_corrected"].notna()
        )
        sub = df[mask]

        if len(sub) == 0:
            raise ValueError(
                f"{label} 대표점 ({center_date.date()}) 근방 "
                f"±{self.window}일 이내 유효 데이터 없음"
            )

        return {
            "date":  str(center_date.date()),
            "theta": float(sub["theta_field"].mean()),
            "N":     float(sub["N_corrected"].mean()),
            "n":     len(sub),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # VWC 계산
    # ─────────────────────────────────────────────────────────────────────────

    def _vwc(self, N: np.ndarray, N0: float, a2: float) -> np.ndarray:
        """
        θ = a0 / (N/N0 − a1) − a2

        a2는 역산된 SHP 당량 (Standard의 Wlat 대신 사용).
        """
        a0, a1 = self.A0, self.A1
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = N / N0 - a1
            vwc   = np.where(denom > 0, a0 / denom - a2, np.nan)
        return np.clip(np.asarray(vwc, dtype=float), 0.0, 1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # 시각화
    # ─────────────────────────────────────────────────────────────────────────

    def plot_result(self,
                    matched_df: pd.DataFrame,
                    result_dir: Path) -> Path:
        """
        [1] calibration_timeseries.png  — FDR vs CRNP VWC + 대표점 표시
        [2] shp_curve.png               — Desilets 곡선 (a2=0 vs a2=역산값 비교)
        """
        if self.result is None:
            raise RuntimeError("calibrate() 를 먼저 실행하세요.")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        res = self.result
        m   = res.metrics
        out_dir = result_dir / self.station_id / "shp_2pt"
        out_dir.mkdir(parents=True, exist_ok=True)

        df = matched_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        N_all    = df["N_corrected"].values.astype(float)
        df["VWC"] = self._vwc(N_all, res.N0, res.a2)
        eval_mask = ~df["date"].dt.month.isin(self.exclude_months)

        # ══ Plot 1: 시계열 ════════════════════════════════════════════════
        fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                                 gridspec_kw={"height_ratios": [3, 1]},
                                 sharex=True)
        ax_main, ax_diff = axes

        if self.candidate_start and self.candidate_end:
            ax_main.axvspan(
                pd.to_datetime(self.candidate_start),
                pd.to_datetime(self.candidate_end),
                color="lightyellow", alpha=0.5,
                label=f"Candidate period ({self.candidate_start} ~ {self.candidate_end})",
                zorder=0,
            )
        _shade_winter(ax_main, df["date"], self.exclude_months)

        ax_main.plot(df["date"], df["theta_field"],
                     color="steelblue", linewidth=1.2,
                     label=r"FDR $\theta_{field}$", zorder=3)
        ax_main.plot(df[eval_mask]["date"], df[eval_mask]["VWC"],
                     color="tomato", linewidth=1.4,
                     label="CRNP VWC (eval)", zorder=4)
        ax_main.plot(df[~eval_mask]["date"], df[~eval_mask]["VWC"],
                     color="tomato", linewidth=0.8, linestyle="--", alpha=0.45,
                     label="CRNP VWC (excluded)", zorder=4)

        # 대표점 마커
        dry = res.extra.get("dry_point", {})
        wet = res.extra.get("wet_point", {})
        if dry and wet:
            ax_main.axvline(pd.to_datetime(dry["date"]),
                            color="saddlebrown", linewidth=1.6, linestyle=":",
                            label=f"Dry point ({dry['date']})")
            ax_main.axvline(pd.to_datetime(wet["date"]),
                            color="navy", linewidth=1.6, linestyle=":",
                            label=f"Wet point ({wet['date']})")

        txt = (f"a2   = {res.a2:.4f}  (SHP)\n"
               f"N0   = {res.N0:.1f}\n"
               f"─────────────\n"
               f"RMSE = {m['RMSE']:.4f}\n"
               f"R    = {m['R']:.3f}\n"
               f"NSE  = {m['NSE']:.3f}\n"
               f"n    = {m['n']}")
        ax_main.text(0.01, 0.97, txt, transform=ax_main.transAxes,
                     va="top", ha="left", fontsize=9, fontfamily="monospace",
                     bbox=dict(boxstyle="round,pad=0.4", fc="white",
                               alpha=0.85, ec="gray"))

        ax_main.set_ylabel("VWC (m³/m³)", fontsize=11)
        ax_main.set_ylim(0, 0.6)
        ax_main.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
        ax_main.set_title(
            f"[{self.station_id}] SHP 2-Point Calibration  —  "
            f"FDR $\\theta_{{field}}$ vs CRNP VWC\n"
            f"a2 = {res.a2:.4f}  |  N0 = {res.N0:.1f}  |  "
            f"RMSE = {m['RMSE']:.4f}  |  R = {m['R']:.3f}",
            fontsize=11,
        )
        ax_main.grid(axis="y", linestyle="--", alpha=0.4)

        diff = df["VWC"] - df["theta_field"]
        ax_diff.bar(df["date"], diff,
                    color=np.where(diff.fillna(0) >= 0, "tomato", "steelblue"),
                    width=1.5, alpha=0.7)
        ax_diff.axhline(0, color="black", linewidth=0.8)
        ax_diff.set_ylabel("Residual\n(VWC − θ)", fontsize=9)
        ax_diff.set_ylim(-0.3, 0.3)
        ax_diff.grid(axis="y", linestyle="--", alpha=0.4)
        ax_diff.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax_diff.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax_diff.xaxis.get_majorticklabels(), rotation=30, ha="right")
        plt.tight_layout()

        p1 = out_dir / "calibration_timeseries.png"
        fig.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  📊 [1] {p1.name}")

        # ══ Plot 2: Desilets 곡선 비교 ════════════════════════════════════
        fig2, ax = plt.subplots(figsize=(8, 6))
        theta_range = np.linspace(0.01, 0.6, 300)
        a0, a1 = 0.0808, 0.372

        for a2_val, label, color, ls in [
            (0.0,    f"Standard (a2=0.000)",         "gray",   "--"),
            (0.115,  f"Standard (a2=0.115, default)", "orange", "-."),
            (res.a2, f"SHP-2pt  (a2={res.a2:.4f})",  "tomato", "-"),
        ]:
            N_curve = res.N0 * (a0 / (theta_range + a2_val) + a1)
            ax.plot(theta_range, N_curve, color=color, linestyle=ls,
                    linewidth=2, label=label)

        # 건조·습윤 대표점
        if dry and wet:
            ax.scatter([dry["theta"], wet["theta"]],
                       [dry["N"],    wet["N"]],
                       color=["saddlebrown", "navy"],
                       s=120, zorder=5,
                       label="Dry / Wet ref. points")

        # 전체 관측 산점도
        valid = df[df["N_corrected"].notna() & df["theta_field"].notna()]
        ax.scatter(valid["theta_field"], valid["N_corrected"],
                   color="steelblue", alpha=0.15, s=10, label="Observations")

        ax.set_xlabel(r"$\theta_{field}$ (m³/m³)", fontsize=11)
        ax.set_ylabel("N_corrected (counts)", fontsize=11)
        ax.set_title(f"[{self.station_id}] Desilets Calibration Curves\n"
                     f"(N0={res.N0:.1f})", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(linestyle="--", alpha=0.4)
        ax.set_xlim(0, 0.65)
        plt.tight_layout()

        p2 = out_dir / "shp_curve.png"
        fig2.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  📊 [2] {p2.name}")

        return p1

    # ─────────────────────────────────────────────────────────────────────────
    # 결과 저장
    # ─────────────────────────────────────────────────────────────────────────

    def save_result(self, result_dir: Path) -> Path:
        if self.result is None:
            raise RuntimeError("calibrate() 를 먼저 실행하세요.")
        out_dir = result_dir / self.station_id / "shp_2pt"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.result.save(result_dir)
        print(f"\n  💾 결과 저장 → {out_dir}")
        return out_dir


# ═══════════════════════════════════════════════════════════════════════════════
# 모듈 레벨 헬퍼
# ═══════════════════════════════════════════════════════════════════════════════

def _quick_metrics(obs: np.ndarray, vwc: np.ndarray) -> Dict:
    valid = np.isfinite(obs) & np.isfinite(vwc)
    if valid.sum() < 2:
        return {"RMSE": np.nan, "R": np.nan, "Bias": np.nan, "n": 0}
    o, v = obs[valid], vwc[valid]
    rmse = float(np.sqrt(np.mean((v - o) ** 2)))
    bias = float(np.mean(v - o))
    r    = float(np.corrcoef(o, v)[0, 1])
    return {"RMSE": rmse, "R": r, "Bias": bias, "n": int(valid.sum())}


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
                   alpha=0.35, zorder=1,
                   label=f"Excluded months {exclude_months}")