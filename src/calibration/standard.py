# src/calibration/standard.py
"""
StandardCalibrator  ──  Desilets 방식 N0 교정
==============================================

알고리즘:
  1. 후보 기간(calibration.candidate_start ~ end) 내
     exclude_months 제외한 날짜를 후보로 추출
  2. 각 후보 날짜의 단일 (θ_field, N_corrected) 로 N0 역산
       minimize  [VWC(N_cal, N0) - θ_field]²
  3. 역산한 N0를 평가 기간(exclude_months 제외) 전체에 적용
       CRNP VWC vs θ_field  →  RMSE 계산
  4. RMSE 최소 날짜 = 최적 교정 날짜, 해당 N0 채택

파라미터 출처 (processing_options.yaml):
  calibration.candidate_start / candidate_end
  calibration.exclude_months
  calibration.standard.wlat       (null → 0.0)
  calibration.standard.wsoc
  calibration.standard.initial_N0
  calibration.standard.n0_bounds
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseCalibrator, CalibrationResult


class StandardCalibrator(BaseCalibrator):
    """
    Parameters
    ----------
    station_cfg : dict   (YAML station 섹션)
    options     : dict   (YAML processing_options 섹션)
    """

    def __init__(self, station_cfg: Dict, options: Dict):
        super().__init__(station_cfg, options)

        cal = options.get("calibration", {})
        std = cal.get("standard", {})

        self.candidate_start: str       = cal.get("candidate_start", "")
        self.candidate_end:   str       = cal.get("candidate_end",   "")
        self.exclude_months:  List[int] = cal.get("exclude_months",  [11, 12, 1, 2])

        # Wlat: YAML에 명시되면 사용, null이면 0.0
        wlat_yaml = std.get("wlat", None)
        self.wlat:   float = float(wlat_yaml) if wlat_yaml is not None else 0.0
        self.wsoc:   float = float(std.get("wsoc",       0.01))
        self.init_N0: float = float(std.get("initial_N0", 1000))
        bounds = std.get("n0_bounds", [300, 5000])
        self.n0_bounds = (float(bounds[0]), float(bounds[1]))

    # ─────────────────────────────────────────────────────────────────────────
    # 공개 인터페이스
    # ─────────────────────────────────────────────────────────────────────────

    def calibrate(self, matched_df: pd.DataFrame) -> CalibrationResult:
        """
        Parameters
        ----------
        matched_df : columns [date, theta_field, N_corrected]
                     DataMatcher.run() 의 출력

        Returns
        -------
        CalibrationResult
        """
        print(f"\n{'='*60}")
        print(f"  StandardCalibrator  ─  {self.station_id}")
        print(f"  Wlat={self.wlat}  Wsoc={self.wsoc}  "
              f"bulk_density={self.bulk_density}")
        print(f"  후보 기간: {self.candidate_start} ~ {self.candidate_end}")
        print(f"  제외 월: {self.exclude_months}")
        print(f"{'='*60}")

        df = matched_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # ── 후보 날짜 필터링 ──────────────────────────────────────────────
        candidates = self._filter_candidates(df)
        if len(candidates) == 0:
            raise ValueError(
                f"유효한 후보 날짜 없음. "
                f"candidate_start/end 및 exclude_months 설정을 확인하세요."
            )

        # ── 평가 데이터: exclude_months 제외 전체 기간 ──────────────────
        eval_df = self._filter_eval(df)
        if len(eval_df) < 10:
            raise ValueError(f"평가 데이터 부족: {len(eval_df)}일")

        print(f"\n  후보 날짜: {len(candidates)}개")
        print(f"  평가 데이터: {len(eval_df)}일")

        # ── 날짜 탐색 ────────────────────────────────────────────────────
        best = self._search_best_date(candidates, eval_df)

        print(f"\n  ✅ 최적 교정 날짜: {best['date'].date()}")
        print(f"     N0   = {best['N0']:.2f}")
        print(f"     RMSE = {best['RMSE']:.4f}")
        print(f"     R    = {best['R']:.3f}")

        # ── 최적 N0로 평가 기간 전체 VWC 산출 ───────────────────────────
        N_eval   = eval_df["N_corrected"].values.astype(float)
        obs_eval = eval_df["theta_field"].values.astype(float)
        vwc_eval = self._vwc(N_eval, best["N0"])

        result = CalibrationResult(
            method     = "standard",
            station_id = self.station_id,
            N0         = best["N0"],
            a2         = self.wlat,
            vwc        = vwc_eval,
            obs        = obs_eval,
            cal_date   = best["date"].date(),
            extra      = {
                "Wlat":           self.wlat,
                "Wsoc":           self.wsoc,
                "bulk_density":   self.bulk_density,
                "candidate_start": self.candidate_start,
                "candidate_end":   self.candidate_end,
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
        """
        후보 기간 + exclude_months 제외 + theta_field/N_corrected 모두 유효한 행.
        """
        mask = pd.Series(True, index=df.index)

        if self.candidate_start:
            mask &= df["date"] >= pd.to_datetime(self.candidate_start)
        if self.candidate_end:
            mask &= df["date"] <= pd.to_datetime(self.candidate_end)

        mask &= ~df["date"].dt.month.isin(self.exclude_months)
        mask &= df["theta_field"].notna()
        mask &= df["N_corrected"].notna()

        return df[mask].copy()

    def _filter_eval(self, df: pd.DataFrame) -> pd.DataFrame:
        """exclude_months 제외 + 양쪽 값 모두 유효한 행."""
        mask = (
            ~df["date"].dt.month.isin(self.exclude_months)
            & df["theta_field"].notna()
            & df["N_corrected"].notna()
        )
        return df[mask].copy()

    def _search_best_date(self,
                          candidates: pd.DataFrame,
                          eval_df:    pd.DataFrame) -> Dict:
        """
        모든 후보 날짜를 순회하며 RMSE 최소 날짜를 찾는다.

        Returns
        -------
        dict: date, N0, RMSE, R, all_metrics
        """
        N_eval   = eval_df["N_corrected"].values.astype(float)
        obs_eval = eval_df["theta_field"].values.astype(float)

        best_rmse = np.inf
        best_N0   = None
        best_date = None
        all_metrics: List[Dict] = []

        total = len(candidates)
        print(f"\n  날짜 탐색 중... ({total}개 후보)")

        for i, (_, row) in enumerate(candidates.iterrows(), 1):
            theta_cal = float(row["theta_field"])
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

            # 진행 상황 (100개마다 출력)
            if i % 100 == 0:
                print(f"    {i}/{total} 완료  현재 best RMSE={best_rmse:.4f}")

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
        단일 (θ, N) 쌍으로 N0 해석적 역산.

        Desilets 식:
          N/N0 = a0/(θ + a2) + a1
          N0   = N / (a0/(θ + a2) + a1)

        minimize_scalar 대신 해석적 풀이를 사용:
          - 클리핑으로 인한 degenerate solution (N0=N/a1) 문제 없음
          - bounds 외 값은 None 반환
        """
        try:
            a2    = self.wlat   # Standard 방식에서 a2 고정값
            denom = self.A0 / (theta + a2) + self.A1
            if denom <= 0 or theta + a2 <= 0:
                return None
            N0 = N / denom
            lo, hi = self.n0_bounds
            return float(N0) if lo <= N0 <= hi else None
        except Exception:
            return None

    def _vwc(self, N: np.ndarray, N0: float) -> np.ndarray:
        """
        N_corrected → VWC.
        crnpy 사용 가능하면 counts_to_vwc(), 아니면 Desilets 직접 계산.
        """
        try:
            import crnpy
            vwc = crnpy.counts_to_vwc(
                N, N0=N0,
                bulk_density=self.bulk_density,
                Wlat=self.wlat,
                Wsoc=self.wsoc,
            )
        except Exception:
            # fallback: θ = a0 / (N/N0 - a1) - a2
            denom = N / N0 - self.A1
            with np.errstate(divide="ignore", invalid="ignore"):
                vwc = np.where(denom > 0, self.A0 / denom - self.wlat, np.nan)

        return np.clip(np.asarray(vwc, dtype=float), 0.0, 1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # 시각화
    # ─────────────────────────────────────────────────────────────────────────

    def plot_result(self,
                    matched_df: pd.DataFrame,
                    result_dir: Path) -> Path:
        """
        두 가지 플롯을 생성하여 저장:
          [1] calibration_timeseries.png  — FDR vs CRNP VWC
          [2] diagnostic_N_corrected.png  — N_corrected 원시 시계열 (드리프트 진단용)
        """
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

        N_all    = df["N_corrected"].values.astype(float)
        df["VWC"] = self._vwc(N_all, res.N0)
        eval_mask = ~df["date"].dt.month.isin(self.exclude_months)

        # ══════════════════════════════════════════════════════════════════
        # Plot 1: FDR θ_field vs CRNP VWC
        # ══════════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(
            2, 1, figsize=(16, 10),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
        ax_main, ax_diff = axes

        # 배경: 후보 기간
        if self.candidate_start and self.candidate_end:
            ax_main.axvspan(
                pd.to_datetime(self.candidate_start),
                pd.to_datetime(self.candidate_end),
                color="lightyellow", alpha=0.5,
                label=f"Candidate period ({self.candidate_start} ~ {self.candidate_end})",
                zorder=0,
            )

        # 동절기 음영
        _shade_winter(ax_main, df["date"], self.exclude_months)

        # FDR θ_field
        ax_main.plot(
            df["date"], df["theta_field"],
            color="steelblue", linewidth=1.2,
            label="FDR $\\theta_{field}$ (distance-weighted)",
            zorder=3,
        )

        # CRNP VWC — 평가 기간(실선) / 제외 기간(점선)
        df_eval    = df[eval_mask]
        df_noneval = df[~eval_mask]
        ax_main.plot(
            df_eval["date"], df_eval["VWC"],
            color="tomato", linewidth=1.4,
            label="CRNP VWC (eval period)",
            zorder=4,
        )
        ax_main.plot(
            df_noneval["date"], df_noneval["VWC"],
            color="tomato", linewidth=0.8, linestyle="--", alpha=0.45,
            label="CRNP VWC (excluded period)",
            zorder=4,
        )

        # 교정 날짜 수직선
        ax_main.axvline(
            pd.to_datetime(res.cal_date),
            color="darkgreen", linewidth=1.8, linestyle="-.", zorder=5,
            label=f"Cal. date: {res.cal_date}",
        )

        # 메트릭 텍스트 박스
        txt = (f"N0   = {res.N0:.1f}\n"
               f"Wlat = {res.a2:.4f}\n"
               f"─────────────\n"
               f"RMSE = {m['RMSE']:.4f}\n"
               f"R    = {m['R']:.3f}\n"
               f"NSE  = {m['NSE']:.3f}\n"
               f"n    = {m['n']}")
        ax_main.text(
            0.01, 0.97, txt,
            transform=ax_main.transAxes,
            va="top", ha="left", fontsize=9,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85, ec="gray"),
        )

        ax_main.set_ylabel("VWC (m³/m³)", fontsize=11)
        ax_main.set_ylim(0, 0.6)
        ax_main.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax_main.set_title(
            f"[{self.station_id}] Standard Calibration  —  "
            f"FDR $\\theta_{{field}}$ vs CRNP VWC\n"
            f"Cal. date: {res.cal_date}  |  "
            f"N0 = {res.N0:.1f}  |  RMSE = {m['RMSE']:.4f}  |  "
            f"R = {m['R']:.3f}  |  NSE = {m['NSE']:.3f}",
            fontsize=11,
        )
        ax_main.grid(axis="y", linestyle="--", alpha=0.4)

        # 하단: 잔차 bar
        diff = df["VWC"] - df["theta_field"]
        colors = np.where(diff.fillna(0) >= 0, "tomato", "steelblue")
        ax_diff.bar(df["date"], diff, color=colors, width=1.5, alpha=0.7)
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

        # ══════════════════════════════════════════════════════════════════
        # Plot 2: N_corrected 진단 시계열 (드리프트 / 스파이크 확인)
        # ══════════════════════════════════════════════════════════════════
        fig2, axes2 = plt.subplots(
            3, 1, figsize=(16, 11),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1, 1]},
        )
        ax_n, ax_vwc2, ax_fdr = axes2

        # N_corrected 원시 시계열
        ax_n.plot(df["date"], df["N_corrected"],
                  color="dimgray", linewidth=0.9, label="N_corrected (daily mean)")

        # N_corrected 30일 이동 평균 (드리프트 강조)
        n_roll = df.set_index("date")["N_corrected"].rolling("30D", min_periods=10).mean()
        ax_n.plot(n_roll.index, n_roll.values,
                  color="black", linewidth=2.0, linestyle="-",
                  label="30-day rolling mean", zorder=5)

        ax_n.set_ylabel("N_corrected\n(counts/day)", fontsize=9)
        ax_n.legend(fontsize=8)
        ax_n.grid(axis="y", linestyle="--", alpha=0.4)
        ax_n.set_title(
            f"[{self.station_id}] Diagnostic — N_corrected / CRNP VWC / FDR θ_field",
            fontsize=11,
        )
        _shade_winter(ax_n, df["date"], self.exclude_months)

        # CRNP VWC
        ax_vwc2.plot(df["date"], df["VWC"],
                     color="tomato", linewidth=0.9, label="CRNP VWC")
        vwc_roll = df.set_index("date")["VWC"].rolling("30D", min_periods=10).mean()
        ax_vwc2.plot(vwc_roll.index, vwc_roll.values,
                     color="darkred", linewidth=2.0, label="30-day rolling mean")
        ax_vwc2.set_ylabel("CRNP VWC\n(m³/m³)", fontsize=9)
        ax_vwc2.set_ylim(0, 0.7)
        ax_vwc2.legend(fontsize=8)
        ax_vwc2.grid(axis="y", linestyle="--", alpha=0.4)
        _shade_winter(ax_vwc2, df["date"], self.exclude_months)

        # FDR θ_field
        ax_fdr.plot(df["date"], df["theta_field"],
                    color="steelblue", linewidth=0.9,
                    label="FDR $\\theta_{field}$")
        fdr_roll = df.set_index("date")["theta_field"].rolling("30D", min_periods=10).mean()
        ax_fdr.plot(fdr_roll.index, fdr_roll.values,
                    color="darkblue", linewidth=2.0, label="30-day rolling mean")
        ax_fdr.set_ylabel("FDR θ_field\n(m³/m³)", fontsize=9)
        ax_fdr.set_ylim(0, 0.6)
        ax_fdr.legend(fontsize=8)
        ax_fdr.grid(axis="y", linestyle="--", alpha=0.4)
        _shade_winter(ax_fdr, df["date"], self.exclude_months)

        ax_fdr.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax_fdr.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax_fdr.xaxis.get_majorticklabels(), rotation=30, ha="right")

        plt.tight_layout()
        p2 = out_dir / "diagnostic_N_corrected.png"
        fig2.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  📊 [2] {p2.name}")

        return p1

    # ─────────────────────────────────────────────────────────────────────────
    # 결과 저장
    # ─────────────────────────────────────────────────────────────────────────

    def save_result(self, result_dir: Path) -> Path:
        """
        교정 결과를 JSON + Excel 로 저장.

        결과 경로:
          {result_dir}/{station_id}/standard/calibration_result.json
          {result_dir}/{station_id}/standard/all_metrics.xlsx
        """
        if self.result is None:
            raise RuntimeError("calibrate() 를 먼저 실행하세요.")

        out_dir = result_dir / self.station_id / "standard"
        out_dir.mkdir(parents=True, exist_ok=True)

        # JSON (CalibrationResult.save)
        self.result.save(result_dir)

        # 날짜별 전체 메트릭 Excel
        metrics = self.result.extra.get("all_metrics", [])
        if metrics:
            df_m = pd.DataFrame(metrics)
            df_m.to_excel(out_dir / "all_metrics.xlsx", index=False)

        print(f"\n  💾 결과 저장 → {out_dir}")
        print(f"     calibration_result.json")
        if metrics:
            print(f"     all_metrics.xlsx  ({len(metrics)}개 후보)")

        return out_dir


# ═══════════════════════════════════════════════════════════════════════════════
# 모듈 레벨 헬퍼
# ═══════════════════════════════════════════════════════════════════════════════

def _shade_winter(ax, dates: pd.Series, exclude_months: List[int]) -> None:
    """
    exclude_months 에 해당하는 기간을 회색 배경으로 표시.
    연도 경계를 처리하기 위해 연속 구간을 찾아서 axvspan.
    """
    import matplotlib.pyplot as plt

    is_excluded = dates.dt.month.isin(exclude_months)
    if not is_excluded.any():
        return

    # 연속 구간 찾기
    in_block  = False
    block_start = None
    for date, excl in zip(dates, is_excluded):
        if excl and not in_block:
            block_start = date
            in_block    = True
        elif not excl and in_block:
            ax.axvspan(block_start, date,
                       color="lightgray", alpha=0.35, zorder=1)
            in_block = False
    if in_block:
        ax.axvspan(block_start, dates.iloc[-1],
                   color="lightgray", alpha=0.35, zorder=1,
                   label=f"제외 기간 (월: {exclude_months})")