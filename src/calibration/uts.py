# src/calibration/uts.py
"""
UTSCalibrator  ──  Universal Transfer Standard (UTS) 교정
==========================================================

[원본: UTS_HC.py / UTS_HC2.py]

[UTS vs Standard 핵심 차이]
  Standard (Desilets 2010):
    θ = a0/(N/N0 − a1) − a2
    → 절대습도(h)를 fw 보정으로 사전 제거한 N_corrected 사용
    → N0 1개 교정

  UTS:
    N = ND × I_norm(θ, h, p)
    → 절대습도(h)를 I_norm 안에 직접 포함 → N_raw(또는 fp/fi 보정만 된 값) 사용
    → ND 1개 교정
    → 4가지 파라미터 셋 (MCNP/URANOS × drf/THL)

[I_norm 식]
  I_norm(θ, h, p) = ((p1+p2·θ)/(p1+θ)) × (p0 + p6·h + p7·h²)
                  + exp(-p3·θ) × (p4 + p5·h)

[ND 역산 (교정 날짜)]
  ND = median( N_cal / I_norm(θ_cal, h_cal, p) )

[VWC 역산 (전 기간)]
  N_i = ND × I_norm(θ, h_i, p)  →  θ_i via bisection

[교정 절차]
  1. 후보 날짜마다 ND 역산
  2. 그 ND로 평가 기간 전체 VWC 계산 → RMSE(VWC, FDR)
  3. RMSE 최소 날짜 = 최적 교정 날짜

[입력 컬럼] matched_df:
  date | theta_field | fdr_avg | N_corrected | abs_humidity
  ※ abs_humidity 단위: g/m³

[UTS 파라미터 셋] YAML calibration.uts.parameter_sets
  4가지 셋 (p0~p7):
    MCNP_drf, MCNP_THL, URANOS_drf, URANOS_THL
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseCalibrator, CalibrationResult


# ── 기본 파라미터 셋 (UTS_HC.py 원본) ────────────────────────────────────────
_DEFAULT_PARAMETER_SETS: Dict[str, Tuple] = {
    "MCNP_drf":   (1.0940, 0.0280, 0.254, 3.537, 0.139, -0.00140, -0.0088, 0.0001150),
    "MCNP_THL":   (1.2650, 0.0259, 0.135, 1.237, 0.063, -0.00021, -0.0117, 0.0001200),
    "URANOS_drf": (1.0240, 0.0226, 0.207, 1.625, 0.235, -0.00290, -0.0093, 0.0000740),
    "URANOS_THL": (1.2230, 0.0185, 0.142, 2.568, 0.155, -0.00047, -0.0119, 0.0000920),
}

_THETA_MIN: float = 0.00
_THETA_MAX: float = 1.00


# ── 핵심 수식 함수 ────────────────────────────────────────────────────────────

def I_norm(theta: np.ndarray,
           h:     np.ndarray,
           p:     Tuple) -> np.ndarray:
    """
    UTS 정규화 중성자 강도.

    I(θ, h) = ((p1+p2·θ)/(p1+θ)) × (p0 + p6·h + p7·h²)
            + exp(-p3·θ) × (p4 + p5·h)

    Parameters
    ----------
    theta : VWC array
    h     : 절대습도 [g/m³] array
    p     : (p0, p1, p2, p3, p4, p5, p6, p7)
    """
    p0, p1, p2, p3, p4, p5, p6, p7 = p
    return (
        ((p1 + p2 * theta) / (p1 + theta)) * (p0 + p6 * h + p7 * h * h)
        + np.exp(-p3 * theta) * (p4 + p5 * h)
    )


def invert_theta_bisect(N_meas: float,
                        h:      float,
                        ND:     float,
                        p:      Tuple,
                        lo:     float = _THETA_MIN,
                        hi:     float = _THETA_MAX,
                        tol:    float = 1e-10,
                        it:     int   = 80) -> float:
    """
    N = ND × I_norm(θ, h, p)  →  θ  (bisection).
    해가 없으면 nan 반환.
    """
    def f(t):
        return ND * I_norm(np.array([t]), np.array([h]), p)[0] - N_meas

    flo, fhi = f(lo), f(hi)
    if not (np.isfinite(flo) and np.isfinite(fhi)):
        return np.nan
    if flo * fhi > 0:
        return np.nan

    for _ in range(it):
        mid = 0.5 * (lo + hi)
        fm  = f(mid)
        if abs(fm) < tol or (hi - lo) < tol:
            return mid
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


# ═══════════════════════════════════════════════════════════════════════════════
# UTSCalibrator
# ═══════════════════════════════════════════════════════════════════════════════

class UTSCalibrator(BaseCalibrator):

    def __init__(self, station_cfg: Dict, options: Dict):
        super().__init__(station_cfg, options)

        cal = options.get("calibration", {})
        uts = cal.get("uts", {})

        # 교정 기간
        self.cal_start, self.cal_end = self.get_cal_period("uts")

        # 제외 월
        _global_excl = cal.get("exclude_months", [11, 12, 1, 2, 3])
        _local_excl  = uts.get("exclude_months", None)
        self.exclude_months: List[int] = (
            _local_excl if _local_excl is not None else _global_excl
        )

        # RMSE 비교 대상
        self.rmse_target: str = uts.get("rmse_target", "theta_field")

        # ND 집계 방식
        self.nd_agg: str = uts.get("nd_agg", "median")  # "median" or "mean"

        # 파라미터 셋 (YAML 또는 기본값)
        yaml_sets = uts.get("parameter_sets", None)
        if yaml_sets:
            self.parameter_sets: Dict[str, Tuple] = {
                name: tuple(vals) for name, vals in yaml_sets.items()
            }
        else:
            self.parameter_sets = _DEFAULT_PARAMETER_SETS.copy()

    # ─────────────────────────────────────────────────────────────────────────
    # 공개 인터페이스
    # ─────────────────────────────────────────────────────────────────────────

    def calibrate(self, matched_df: pd.DataFrame) -> CalibrationResult:
        """
        4개 파라미터 셋 전부 교정 후
        RMSE 가장 낮은 셋의 결과를 CalibrationResult로 반환.
        개별 셋 결과는 result.extra["sets"]에 모두 저장.
        """
        print(f"\n{'='*60}")
        print(f"  UTSCalibrator  ─  {self.station_id}")
        print(f"  교정 기간  : {self.cal_start or '전체'} ~ {self.cal_end or '전체'}")
        print(f"  제외 월    : {self.exclude_months}")
        print(f"  RMSE 대상  : {self.rmse_target}")
        print(f"  ND 집계    : {self.nd_agg}")
        print(f"  파라미터 셋: {list(self.parameter_sets.keys())}")
        print(f"{'='*60}")

        df = matched_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # abs_humidity 단위 자동 교정 (g/m³ 이어야 함)
        h_raw = df["abs_humidity"].values.astype(float)
        h = self._fix_humidity_unit(h_raw)
        df["abs_humidity_gm3"] = h

        # N_uts: fp·fi만 보정된 중성자 (fw 미적용) — UTS I_norm이 h를 내부 처리
        # N_corrected(fw 포함) 대신 N_uts 사용해야 습도 이중 처리 방지
        if "N_uts" in df.columns:
            N_arr = df["N_uts"].values.astype(float)
            print(f"  ℹ️  N_uts 사용 (fw 미보정, I_norm 내부 처리)")
        else:
            # 구형 데이터 호환: N_corrected에서 fw 역산
            if "fw" in df.columns:
                fw = df["fw"].values.astype(float)
                N_arr = (df["N_corrected"].values.astype(float)
                         / np.where(fw > 0, fw, np.nan))
                print(f"  ℹ️  N_corrected / fw → N_uts 복원")
            else:
                N_arr = df["N_corrected"].values.astype(float)
                print(f"  ⚠️  N_uts/fw 없음 → N_corrected 사용 (습도 이중 보정 주의)")

        # 필요 배열
        theta_arr = df[self.rmse_target].values.astype(float)

        # 유효행 마스크
        valid = (df["date"].notna()
                 & np.isfinite(theta_arr)
                 & np.isfinite(h)
                 & np.isfinite(N_arr))
        df_v      = df[valid].reset_index(drop=True)
        theta_v   = theta_arr[valid]
        h_v       = h[valid]
        N_v       = N_arr[valid]

        if len(df_v) == 0:
            raise ValueError("유효한 데이터 없음 (theta / abs_humidity / N_corrected 확인)")

        # 후보 / 평가 마스크
        cand_mask = self._candidate_mask(df_v)
        eval_mask = self._eval_mask(df_v)

        if cand_mask.sum() == 0:
            raise ValueError("유효한 후보 날짜 없음. calibration_start/end 확인.")
        if eval_mask.sum() < 10:
            raise ValueError(f"평가 데이터 부족: {eval_mask.sum()}일")

        print(f"\n  후보 날짜  : {cand_mask.sum()}개")
        print(f"  평가 데이터: {eval_mask.sum()}일")

        # 셋별 교정
        set_results = {}
        best_rmse   = np.inf
        best_name   = None

        for name, p in self.parameter_sets.items():
            print(f"\n  ▶ [{name}]")
            sr = self._calibrate_one_set(
                name, p, df_v, theta_v, h_v, N_v, cand_mask, eval_mask
            )
            set_results[name] = sr
            print(f"     cal_date={sr['cal_date']}  "
                  f"ND={sr['ND']:.2f}  "
                  f"RMSE={sr['RMSE']:.4f}  "
                  f"R={sr['R']:.3f}")

            if sr["RMSE"] < best_rmse:
                best_rmse = sr["RMSE"]
                best_name = name

        print(f"\n  ✅ 최적 셋: {best_name}  (RMSE={best_rmse:.4f})")

        best = set_results[best_name]

        result = CalibrationResult(
            method     = "uts",
            station_id = self.station_id,
            N0         = best["ND"],          # N0 필드에 ND 저장
            a2         = 0.0,                 # UTS는 a2 개념 없음
            vwc        = best["vwc_eval"],
            obs        = best["obs_eval"],
            cal_date   = best["cal_date"],
            extra      = {
                "best_set":      best_name,
                "rmse_target":   self.rmse_target,
                "nd_agg":        self.nd_agg,
                "calibration_start": self.cal_start,
                "calibration_end":   self.cal_end,
                "exclude_months":  self.exclude_months,
                "sets":            {
                    name: {k: v for k, v in sr.items()
                           if k not in ("vwc_eval", "obs_eval", "all_metrics")}
                    for name, sr in set_results.items()
                },
                "all_metrics": best["all_metrics"],
            },
        )
        self.result      = result
        self._set_results = set_results   # 시각화용
        self._df_valid    = df_v
        self._arrays      = (theta_v, h_v, N_v)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 단일 셋 교정
    # ─────────────────────────────────────────────────────────────────────────

    def _calibrate_one_set(self,
                           name:       str,
                           p:          Tuple,
                           df_v:       pd.DataFrame,
                           theta_v:    np.ndarray,
                           h_v:        np.ndarray,
                           N_v:        np.ndarray,
                           cand_mask:  np.ndarray,
                           eval_mask:  np.ndarray) -> Dict:
        """단일 파라미터 셋에 대한 날짜 탐색 → 최적 ND 반환."""
        candidate_dates = df_v.loc[cand_mask, "date"].dt.date.unique()

        best_date = None
        best_rmse = np.inf
        best_nd   = np.nan
        all_metrics: List[Dict] = []

        for cal_date in candidate_dates:
            mask_cal = df_v["date"].dt.date == cal_date
            if not mask_cal.any():
                continue

            # ND 역산
            nd = self._invert_ND(
                theta_v[mask_cal], h_v[mask_cal], N_v[mask_cal], p
            )
            if nd is None:
                continue

            # 평가 기간 VWC
            vwc_eval = self._invert_VWC(N_v[eval_mask], h_v[eval_mask], nd, p)
            obs_eval = theta_v[eval_mask]

            valid = np.isfinite(vwc_eval) & np.isfinite(obs_eval)
            if valid.sum() < 10:
                continue

            rmse = float(np.sqrt(np.mean((vwc_eval[valid] - obs_eval[valid]) ** 2)))
            r    = float(np.corrcoef(vwc_eval[valid], obs_eval[valid])[0, 1])

            all_metrics.append({
                "date": str(cal_date),
                "ND":   round(nd,   3),
                "RMSE": round(rmse, 6),
                "R":    round(r,    6),
            })

            if rmse < best_rmse:
                best_rmse = rmse
                best_date = cal_date
                best_nd   = nd

        if best_nd is None or not np.isfinite(best_nd):
            return {"cal_date": None, "ND": np.nan, "RMSE": np.inf, "R": np.nan,
                    "vwc_eval": np.full(eval_mask.sum(), np.nan),
                    "obs_eval": theta_v[eval_mask],
                    "all_metrics": []}

        # 최적 ND로 평가 기간 VWC 재계산
        vwc_eval = self._invert_VWC(N_v[eval_mask], h_v[eval_mask], best_nd, p)
        obs_eval = theta_v[eval_mask]

        valid = np.isfinite(vwc_eval) & np.isfinite(obs_eval)
        r_best = float(np.corrcoef(vwc_eval[valid], obs_eval[valid])[0, 1]) if valid.sum() >= 2 else np.nan

        return {
            "cal_date":    str(best_date),
            "ND":          best_nd,
            "RMSE":        best_rmse,
            "R":           r_best,
            "vwc_eval":    vwc_eval,
            "obs_eval":    obs_eval,
            "all_metrics": all_metrics,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 수식 헬퍼
    # ─────────────────────────────────────────────────────────────────────────

    def _invert_ND(self,
                   theta_cal: np.ndarray,
                   h_cal:     np.ndarray,
                   N_cal:     np.ndarray,
                   p:         Tuple) -> Optional[float]:
        """교정점에서 ND 역산: median(N_cal / I_norm(θ_cal, h_cal, p))"""
        I_cal = I_norm(theta_cal, h_cal, p)
        valid = np.isfinite(I_cal) & (I_cal > 0) & np.isfinite(N_cal)
        if not valid.any():
            return None
        ratios = N_cal[valid] / I_cal[valid]
        nd = (float(np.nanmedian(ratios)) if self.nd_agg == "median"
              else float(np.nanmean(ratios)))
        return nd if np.isfinite(nd) else None

    def _invert_VWC(self,
                    N_arr: np.ndarray,
                    h_arr: np.ndarray,
                    ND:    float,
                    p:     Tuple) -> np.ndarray:
        """전체 기간 VWC 역산 (bisection)."""
        vwc = []
        for n_i, h_i in zip(N_arr, h_arr):
            if not (np.isfinite(n_i) and np.isfinite(h_i)):
                vwc.append(np.nan)
                continue
            g_lo = ND * I_norm(np.array([_THETA_MIN]), np.array([h_i]), p)[0]
            g_hi = ND * I_norm(np.array([_THETA_MAX]), np.array([h_i]), p)[0]
            lo, hi = min(g_lo, g_hi), max(g_lo, g_hi)
            if not (lo - 1e-12 <= n_i <= hi + 1e-12):
                vwc.append(np.nan)
                continue
            theta = invert_theta_bisect(n_i, h_i, ND, p)
            vwc.append(theta)
        return np.array(vwc, dtype=float)

    @staticmethod
    def _fix_humidity_unit(h_raw: np.ndarray) -> np.ndarray:
        """
        abs_humidity 단위 자동 교정 → g/m³.
        UTS_HC.py 원본 로직:
          median < 0.5       → ×1e6 (fraction 단위)
          median > 100       → ÷1000 (mg/m³ 단위)
          그 외              → 그대로 (g/m³)
        """
        mh = float(np.nanmedian(h_raw))
        if 1e-6 < mh < 0.5:
            return h_raw * 1_000_000.0
        elif mh > 100.0:
            return h_raw / 1000.0
        return h_raw.copy()

    # ─────────────────────────────────────────────────────────────────────────
    # 마스크 헬퍼
    # ─────────────────────────────────────────────────────────────────────────

    def _candidate_mask(self, df: pd.DataFrame) -> np.ndarray:
        """교정 기간 마스크 — 파라미터(ND) 결정에 사용."""
        mask = ~df["date"].dt.month.isin(self.exclude_months)
        if self.cal_start:
            mask &= df["date"] >= pd.to_datetime(self.cal_start)
        if self.cal_end:
            mask &= df["date"] <= pd.to_datetime(self.cal_end)
        return mask.values

    def _eval_mask(self, df: pd.DataFrame) -> np.ndarray:
        return (~df["date"].dt.month.isin(self.exclude_months)).values

    # ─────────────────────────────────────────────────────────────────────────
    # 결과 저장
    # ─────────────────────────────────────────────────────────────────────────

    def save_result(self, result_dir: Path) -> Path:
        if self.result is None:
            raise RuntimeError("calibrate() 를 먼저 실행하세요.")

        out_dir = result_dir / self.station_id / "uts"
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = self.result.save(result_dir)

        # 셋별 요약 Excel
        rows = []
        for name, sr in self.result.extra.get("sets", {}).items():
            rows.append({
                "set":      name,
                "cal_date": sr.get("cal_date"),
                "ND":       sr.get("ND"),
                "RMSE":     sr.get("RMSE"),
                "R":        sr.get("R"),
            })
        summary_fname = f"{self.station_id}_uts_summary.xlsx"
        pd.DataFrame(rows).to_excel(out_dir / summary_fname, index=False)

        print(f"\n  💾 결과 저장 → {out_dir}")
        print(f"     {json_path.name}")
        print(f"     {summary_fname}")
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

        out_dir = result_dir / self.station_id / "uts"
        out_dir.mkdir(parents=True, exist_ok=True)

        res     = self.result
        df_v    = self._df_valid
        theta_v, h_v, N_v = self._arrays
        ref_col = self.rmse_target

        # MCNP: blue 계열 / URANOS: green 계열
        # drf: 실선(-) / THL: 파선(--)
        COLOR_MAP = {
            "MCNP_drf":   "steelblue",
            "MCNP_THL":   "steelblue",
            "URANOS_drf": "seagreen",
            "URANOS_THL": "seagreen",
        }
        LS_MAP = {
            "MCNP_drf":   "-",
            "MCNP_THL":   "--",
            "URANOS_drf": "-",
            "URANOS_THL": "--",
        }

        # ── Plot 1: 전체 시계열 (4개 셋 오버레이) ──────────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                                 gridspec_kw={"height_ratios": [3, 1]},
                                 sharex=True)
        ax, ax_d = axes

        _shade_winter(ax, df_v["date"], self.exclude_months)

        ax.plot(df_v["date"], df_v[ref_col],
                color="black", lw=1.4, label=f"FDR ({ref_col})", zorder=5)

        eval_mask = self._eval_mask(df_v)
        best_name = res.extra["best_set"]

        for name, p in self.parameter_sets.items():
            sr    = self._set_results[name]
            if not np.isfinite(sr["ND"]):
                continue
            vwc_all = self._invert_VWC(N_v, h_v, sr["ND"], p)
            color   = COLOR_MAP.get(name, "gray")
            ls      = LS_MAP.get(name, "-")
            lw      = 1.4 if name == best_name else 0.9
            alpha   = 1.0 if name == best_name else 0.7
            label   = (f"{name} ★ RMSE={sr['RMSE']:.4f}"
                       if name == best_name
                       else f"{name} RMSE={sr['RMSE']:.4f}")
            ax.plot(df_v["date"][eval_mask], vwc_all[eval_mask],
                    color=color, lw=lw, ls=ls, alpha=alpha,
                    label=label, zorder=4)

            # 교정 날짜 수직선 표시
            if sr["cal_date"]:
                cal_dt = pd.to_datetime(sr["cal_date"])
                ax.axvline(cal_dt, color=color, lw=0.9, ls=":",
                           alpha=0.8, zorder=3)
                label_text = name.replace("_", "\n")
                ax.text(cal_dt, 0.04, label_text,
                        color=color, fontsize=6, ha="center", va="bottom",
                        transform=ax.get_xaxis_transform())

        ax.set_ylabel("VWC (m³/m³)", fontsize=11)
        ax.set_ylim(0, 0.6)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.set_title(
            f"[{self.station_id}] UTS Calibration  ─  FDR vs CRNP VWC\n"
            f"Best: {best_name}  |  ND={res.N0:.1f}  |  "
            f"RMSE={res.metrics['RMSE']:.4f}  |  R={res.metrics['R']:.3f}",
            fontsize=11)
        ax.grid(axis="y", ls="--", alpha=0.4)

        # Residual (best set 기준)
        best_p   = self.parameter_sets[best_name]
        vwc_best = self._invert_VWC(N_v, h_v, res.N0, best_p)
        diff     = pd.Series(vwc_best - theta_v, index=df_v.index)
        ax_d.bar(df_v["date"], diff,
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
        if "rain" in df_v.columns and df_v["rain"].notna().any():
            add_rain_bars(ax, df_v["date"], df_v["rain"])

        plt.tight_layout()

        p1 = out_dir / f"{self.station_id}_uts_timeseries.png"
        fig.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  📊 {p1.name}")

        # ── Plot 2: 셋별 산점도 (2×2, 정사각형) ──────────────────────────
        # figsize를 정사각형 기반으로 설정 (각 subplot 6×6)
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 12))
        axes2 = axes2.flatten()

        lim = [0.1, 0.5]

        for i, (name, p) in enumerate(self.parameter_sets.items()):
            sr  = self._set_results[name]
            ax2 = axes2[i]
            if not np.isfinite(sr["ND"]):
                ax2.set_visible(False)
                continue
            vwc_all = self._invert_VWC(N_v[eval_mask], h_v[eval_mask], sr["ND"], p)
            obs_all = theta_v[eval_mask]
            valid   = np.isfinite(vwc_all) & np.isfinite(obs_all)
            color  = COLOR_MAP.get(name, "gray")
            marker = "o" if "drf" in name.lower() else "s"
            ax2.scatter(obs_all[valid], vwc_all[valid],
                        color=color, alpha=0.4, s=14, marker=marker)
            ax2.plot(lim, lim, "k--", lw=1)
            ax2.set_xlim(lim)
            ax2.set_ylim(lim)
            ax2.set_aspect("equal", adjustable="box")   # ← 정사각형
            ax2.set_xlabel("FDR θ (m³/m³)", fontsize=10)
            ax2.set_ylabel("CRNP VWC", fontsize=10)
            star = " ★" if name == best_name else ""
            ax2.set_title(f"{name}{star}\nRMSE={sr['RMSE']:.4f}  R={sr['R']:.3f}",
                          fontsize=10)
            ax2.grid(ls="--", alpha=0.4)

        fig2.suptitle(f"[{self.station_id}] UTS Scatter — 4 Parameter Sets",
                      fontsize=13)
        plt.tight_layout()
        p2 = out_dir / f"{self.station_id}_uts_scatter.png"
        fig2.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  📊 {p2.name}")

        # ── 단일 산점도 (best set) ─────────────────────────────────────
        from src.utils.plotting import plot_scatter
        p_scat = out_dir / f"{self.station_id}_uts_calibration_scatter.png"
        plot_scatter(res.obs, res.vwc, "uts", self.station_id, res.metrics, p_scat)

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