# src/calibration/shp_opt.py
"""
SHPOptCalibrator  ──  데이터 주도형 무강우 건조기 슬라이딩 윈도우 기반 a2 추출
=================================================================================

[원본 알고리즘 출처]
  "데이터 주도형 무강우 건조기 슬라이딩 윈도우 기반 a2 추출 알고리즘"
  SHP 추정을 위한 슬라이딩 윈도우 알고리즘.md (2026-04-17)

[핵심 철학]
  동적 수소 풀(DHP: 강우, 이슬, 식생 생장 등)은 항상 a2를 과대평가(+)시킨다.
  따라서 역산된 a2 후보군 중 **하위 백분위수 군집의 median**이
  진정한 기저 수소량(True SHP)이다.  → '최솟값 가설'

[shp_2pt와의 차이]
  shp_2pt:  백분위수로 dry/wet 대표점 선택 → 모든 윈도우 median
  shp_opt:  엄격한 건조기 이벤트 탐색 (무강우 + 단조감소) → 하위 백분위수 군집

[Point 정의]
  Point 1 (θ_wet): 강우 종료 후 rain_buffer_days(≈3일) 경과 시점
                   → FDR 하강 곡선이 안정화된 습윤단
  Point 2 (θ_dry): Point 1으로부터 window_days(≈7일) 뒤
                   → 동일한 시간 간격으로 앙상블 통계 신뢰도 확보

[QC 필터 (둘 다 통과해야 유효)]
  1. 강수 조건: 윈도우 내 강수 0mm (1mm 이상 발생 시 폐기)
     → rain 데이터 없으면 이 조건은 skip
  2. 단조 감소 조건: 매일 θ[k] > θ[k+1] (엄격한 단조감소)
     → 단 하루라도 증가/유지되면 폐기 (미기록 강우, 이슬, 노이즈 간주)

[최종 a2 선택]
  후보 a2들의 하위 a2_low_pct ~ a2_high_pct 구간 median
  → DHP 오염된 상위값을 버리고 진정한 SHP 군집만 선택

[강수 데이터]
  matched_df에 'rain' 컬럼이 있으면 자동 사용.
  없으면 YAML rain_path에서 로드.
  둘 다 없으면 강수 조건 skip (단조감소만 적용).

[YAML 파라미터] calibration.shp_opt:
  calibration_start/end : 교정 기간
  exclude_months        : 제외 월 (null → 공통값)
  season_months         : 추가 계절 격리 (null → 전체, 예: [9,10])
  rain_path             : 강수 CSV 파일 경로 (null → matched_df의 rain 컬럼 사용)
  window_days           : Point1 → Point2 고정 간격 [일] (기본 7)
  rain_buffer_days      : 강우 종료 후 버퍼 [일] (기본 3)
  min_rain_mm           : 폐기 강수 임계값 [mm] (기본 1.0)
  a2_low_pct            : 하위 백분위수 하한 (기본 5)
  a2_high_pct           : 하위 백분위수 상한 (기본 20)
  min_events            : 최소 유효 이벤트 수 (기본 5)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseCalibrator, CalibrationResult
from .shp_2pt import solve_2pt


_DEFAULTS = {
    "window_days":      7,
    "rain_buffer_days": 3,
    "min_rain_mm":      1.0,
    "a2_low_pct":       5,
    "a2_high_pct":      20,
    "min_events":       5,
}


class SHPOptCalibrator(BaseCalibrator):

    def __init__(self, station_cfg: Dict, options: Dict):
        super().__init__(station_cfg, options)

        cal = options.get("calibration", {})
        opt = cal.get("shp_opt", {})

        # 교정 기간
        self.cal_start, self.cal_end = self.get_cal_period("shp_opt")

        # 제외 월
        _global_excl = cal.get("exclude_months", [11, 12, 1, 2, 3])
        _local_excl  = opt.get("exclude_months", None)
        self.exclude_months: List[int] = (
            _local_excl if _local_excl is not None else _global_excl
        )

        # 추가 계절 격리 (exclude_months와 별개 — 이 월만 허용)
        self.season_months: Optional[List[int]] = opt.get("season_months", None)

        # 강수 데이터 경로
        self.rain_path: Optional[str] = opt.get("rain_path", None)

        # 알고리즘 파라미터
        def _g(k): return opt.get(k, _DEFAULTS[k])
        self.window_days:      int   = int(_g("window_days"))
        self.rain_buffer_days: int   = int(_g("rain_buffer_days"))
        self.min_rain_mm:      float = float(_g("min_rain_mm"))
        self.a2_low_pct:       float = float(_g("a2_low_pct"))
        self.a2_high_pct:      float = float(_g("a2_high_pct"))
        self.min_events:       int   = int(_g("min_events"))

        self.rmse_target: str = opt.get("rmse_target", "theta_field")

    # ─────────────────────────────────────────────────────────────────────────
    # 공개 인터페이스
    # ─────────────────────────────────────────────────────────────────────────

    def calibrate(self, matched_df: pd.DataFrame) -> CalibrationResult:
        print(f"\n{'='*60}")
        print(f"  SHPOptCalibrator  ─  {self.station_id}")
        print(f"  교정 기간  : {self.cal_start or '전체'} ~ {self.cal_end or '전체'}")
        print(f"  제외 월    : {self.exclude_months}")
        print(f"  계절 격리  : {self.season_months or '없음'}")
        print(f"  윈도우     : {self.window_days}일  버퍼: {self.rain_buffer_days}일")
        print(f"  a2 선택    : 하위 {self.a2_low_pct}~{self.a2_high_pct}% median")
        print(f"{'='*60}")

        df = matched_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # 강수 데이터 병합
        rain_series = self._load_rain(df)
        if rain_series is not None:
            df["rain"] = df["date"].map(rain_series).fillna(0.0)
            print(f"  ✅ 강수 데이터 로드: {rain_series.notna().sum()}일")
        else:
            df["rain"] = np.nan
            print(f"  ⚠️  강수 데이터 없음 → 단조감소 조건만 적용")

        # 교정 기간 필터
        cal_df  = self._filter_candidates(df)
        eval_df = self.filter_eval(df, self.exclude_months, self.rmse_target)

        if len(cal_df) < self.window_days * 2:
            raise ValueError(f"교정 데이터 부족: {len(cal_df)}일")

        print(f"\n  교정 데이터: {len(cal_df)}일  "
              f"({cal_df['date'].min().date()} ~ {cal_df['date'].max().date()})")

        # ── 건조기 이벤트 탐색 ───────────────────────────────────────────
        events = self._extract_drydown_events(cal_df)

        if len(events) < self.min_events:
            raise ValueError(
                f"유효 이벤트 부족: {len(events)}개 (최소 {self.min_events}개 필요)\n"
                f"  → window_days를 줄이거나 min_events를 낮추거나\n"
                f"    season_months 범위를 넓혀보세요."
            )

        a2_vals = np.array([e["a2"] for e in events])
        N0_vals = np.array([e["N0"] for e in events])

        print(f"\n  유효 이벤트: {len(events)}개")
        print(f"  a2 전체 분포: "
              f"min={a2_vals.min():.4f}  "
              f"median={np.median(a2_vals):.4f}  "
              f"max={a2_vals.max():.4f}  "
              f"std={a2_vals.std():.4f}")

        # ── 최솟값 가설: 하위 백분위수 군집에서 최종 a2 선택 ────────────
        lo = np.percentile(a2_vals, self.a2_low_pct)
        hi = np.percentile(a2_vals, self.a2_high_pct)
        cluster_mask = (a2_vals >= lo) & (a2_vals <= hi)
        cluster = a2_vals[cluster_mask]

        if len(cluster) == 0:
            cluster_mask = a2_vals <= np.percentile(a2_vals, self.a2_high_pct)
            cluster = a2_vals[cluster_mask]

        a2_final = float(np.median(cluster))
        print(f"  a2* (하위 {self.a2_low_pct}~{self.a2_high_pct}% 군집, "
              f"n={len(cluster)}): {a2_final:.4f}")

        # ── N0: 동일한 클러스터 이벤트의 N0 median ───────────────────────
        # 2점 역산에서 (a2_i, N0_i)는 쌍으로 도출됨
        # a2*를 클러스터 이벤트에서 가져왔으므로 N0*도 같은 이벤트에서 가져와야
        # 내부 일관성(self-consistency) 유지
        N0_cluster = N0_vals[cluster_mask]
        N0_final   = float(np.median(N0_cluster))
        print(f"  N0* : {N0_final:.2f}  "
              f"(클러스터 이벤트 {len(N0_cluster)}개의 median, "
              f"range=[{N0_cluster.min():.1f}, {N0_cluster.max():.1f}])")

        # ── 평가 기간 VWC ─────────────────────────────────────────────────
        N_eval   = eval_df["N_corrected"].values.astype(float)
        obs_eval = eval_df[self.rmse_target].values.astype(float)
        vwc_eval = self._vwc(N_eval, N0_final, a2_final)

        valid = np.isfinite(vwc_eval) & np.isfinite(obs_eval)
        rmse  = float(np.sqrt(np.mean((vwc_eval[valid] - obs_eval[valid]) ** 2)))
        r     = float(np.corrcoef(vwc_eval[valid], obs_eval[valid])[0, 1])
        print(f"  평가: RMSE={rmse:.4f}  R={r:.3f}  n={valid.sum()}")

        result = CalibrationResult(
            method     = "shp_opt",
            station_id = self.station_id,
            N0         = N0_final,
            a2         = a2_final,
            vwc        = vwc_eval,
            obs        = obs_eval,
            cal_date   = None,
            extra      = {
                "a2_final":        a2_final,
                "a2_cluster":      cluster.tolist(),
                "a2_all":          a2_vals.tolist(),
                "a2_low_pct":      self.a2_low_pct,
                "a2_high_pct":     self.a2_high_pct,
                "N0_cluster":       N0_cluster.tolist(),
                "N0_method":        "Cluster events median (self-consistent with a2*)",
                "n_events":        len(events),
                "n_cluster":       len(cluster),
                "window_days":     self.window_days,
                "rain_buffer_days": self.rain_buffer_days,
                "calibration_start": self.cal_start,
                "calibration_end":   self.cal_end,
                "exclude_months":  self.exclude_months,
                "season_months":   self.season_months,
                "n_eval_days":     len(eval_df),
                "events": [
                    {k: str(v) if isinstance(v, pd.Timestamp) else v
                     for k, v in e.items()}
                    for e in events
                ],
            },
        )
        self.result  = result
        self._cal_df = cal_df
        self._events = events
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 건조기 이벤트 탐색
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_drydown_events(self, df: pd.DataFrame) -> List[Dict]:
        """
        엄격한 QC를 통과한 7일(window_days) 건조기 이벤트 목록 반환.

        각 이벤트: {date_p1, date_p2, theta1, N1, theta2, N2, a2, N0}
        """
        # 날짜 인덱스로 재구성
        df = df.sort_values("date").reset_index(drop=True)
        df = df.set_index("date")

        # 전체 날짜 범위에서 하루씩 슬라이딩
        dates     = df.index
        n         = len(dates)
        W         = self.window_days
        events: List[Dict] = []
        skip_until = pd.Timestamp("1900-01-01")   # 강우 이후 버퍼 관리

        for i in range(n - W):
            p1_date = dates[i]

            # 강수 버퍼: 최근 rain_buffer_days 이내 강우 여부 확인
            if p1_date < skip_until:
                continue

            # 윈도우 슬라이스
            win_dates = dates[i : i + W + 1]   # Point1 포함 W+1일
            win = df.loc[win_dates]

            # 유효값 존재 확인
            theta = win[self.rmse_target].values.astype(float)
            N     = win["N_corrected"].values.astype(float)
            if not (np.isfinite(theta).all() and np.isfinite(N).all()):
                continue

            # ── QC 1: 강수 조건 ──────────────────────────────────────────
            has_rain_data = "rain" in win.columns and win["rain"].notna().any()
            if has_rain_data:
                rain_in_window = win["rain"].fillna(0).values
                if rain_in_window.max() >= self.min_rain_mm:
                    # 강우 발생: 강우 종료 후 buffer_days skip
                    last_rain_idx = np.where(
                        rain_in_window >= self.min_rain_mm
                    )[0].max()
                    skip_until = win_dates[last_rain_idx] + pd.Timedelta(
                        days=self.rain_buffer_days
                    )
                    continue

            # ── QC 2: 엄격한 단조 감소 조건 ─────────────────────────────
            # θ[0] > θ[1] > ... > θ[W]  (매일 반드시 감소)
            if not np.all(np.diff(theta) < 0):
                continue

            # ── 2점 역산 ─────────────────────────────────────────────────
            t1, N1 = float(theta[0]),  float(N[0])   # Point 1: 습윤단
            t2, N2 = float(theta[-1]), float(N[-1])  # Point 2: 건조단

            a2, N0 = solve_2pt(t1, N1, t2, N2,
                               a2_min=0.0, a2_max=0.5)
            if a2 is None:
                continue

            events.append({
                "date_p1": str(p1_date.date()),
                "date_p2": str(win_dates[-1].date()),
                "theta1":  round(t1, 5),
                "N1":      round(N1, 2),
                "theta2":  round(t2, 5),
                "N2":      round(N2, 2),
                "delta_theta": round(t1 - t2, 5),
                "a2":      round(a2, 6),
                "N0":      round(N0, 3),
            })

        return events

    # ─────────────────────────────────────────────────────────────────────────
    # 강수 데이터 로드
    # ─────────────────────────────────────────────────────────────────────────

    def _load_rain(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """
        강수 데이터 로드. 우선순위:
          1. matched_df의 'rain' 컬럼
          2. YAML rain_path CSV
          3. None (강수 조건 skip)
        """
        # 1. matched_df에 rain 컬럼이 있으면 그대로 사용
        if "rain" in df.columns and df["rain"].notna().any():
            return df.set_index("date")["rain"]

        # 2. YAML rain_path
        if self.rain_path:
            try:
                p = Path(self.rain_path)
                df_rain = pd.read_csv(p, encoding="utf-8-sig")
                # 날짜 컬럼 자동 탐색
                date_col = next(
                    (c for c in df_rain.columns
                     if c.lower() in ("date", "날짜", "일시", "timestamp")),
                    df_rain.columns[0]
                )
                # 강수량 컬럼 자동 탐색
                rain_col = next(
                    (c for c in df_rain.columns
                     if any(k in c.lower()
                            for k in ("rain", "강수", "precipitation", "prcp"))),
                    df_rain.columns[1]
                )
                df_rain[date_col] = pd.to_datetime(df_rain[date_col])
                df_rain[rain_col] = pd.to_numeric(df_rain[rain_col], errors="coerce").fillna(0)
                return df_rain.set_index(date_col)[rain_col]
            except Exception as e:
                print(f"  ⚠️  강수 데이터 로드 실패 ({self.rain_path}): {e}")

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # 필터 / VWC / N0
    # ─────────────────────────────────────────────────────────────────────────

    def _filter_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = pd.Series(True, index=df.index)
        if self.cal_start:
            mask &= df["date"] >= pd.to_datetime(self.cal_start)
        if self.cal_end:
            mask &= df["date"] <= pd.to_datetime(self.cal_end)
        mask &= ~df["date"].dt.month.isin(self.exclude_months)
        # 계절 격리: season_months가 지정되면 해당 월만 허용
        if self.season_months:
            mask &= df["date"].dt.month.isin(self.season_months)
        mask &= df[self.rmse_target].notna()
        mask &= df["N_corrected"].notna()
        return df[mask].copy()

    def _refit_N0(self, df: pd.DataFrame, a2: float) -> float:
        """a2 고정 후 교정 기간 전체 데이터로 N0 재역산 (median)."""
        a0, a1 = self.A0, self.A1
        theta  = df[self.rmse_target].values.astype(float)
        N      = df["N_corrected"].values.astype(float)
        denom  = a0 / (theta + a2) + a1
        valid  = (denom > 0) & np.isfinite(denom) & np.isfinite(N)
        N0_arr = N[valid] / denom[valid]
        N0_arr = N0_arr[(N0_arr > 300) & (N0_arr < 5000)]
        if len(N0_arr) == 0:
            raise ValueError("N0 재역산 실패: 유효 데이터 없음")
        return float(np.median(N0_arr))

    def _vwc(self, N: np.ndarray, N0: float, a2: float) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = N / N0 - self.A1
            vwc   = np.where(denom > 0, self.A0 / denom - a2, np.nan)
        return np.clip(np.asarray(vwc, dtype=float), 0.0, 1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # 결과 저장
    # ─────────────────────────────────────────────────────────────────────────

    def save_result(self, result_dir: Path) -> Path:
        if self.result is None:
            raise RuntimeError("calibrate() 를 먼저 실행하세요.")

        out_dir = result_dir / self.station_id / "shp_opt"
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.result.save(result_dir)

        # 이벤트 목록 Excel
        if self._events:
            ev_fname = f"{self.station_id}_shp_opt_events.xlsx"
            pd.DataFrame(self._events).to_excel(
                out_dir / ev_fname, index=False
            )
            print(f"     {ev_fname}  ({len(self._events)}개 이벤트)")

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
        from src.utils.plotting import plot_scatter

        res     = self.result
        m       = res.metrics
        out_dir = result_dir / self.station_id / "shp_opt"
        out_dir.mkdir(parents=True, exist_ok=True)

        df = matched_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df["VWC"] = self._vwc(df["N_corrected"].values.astype(float),
                              res.N0, res.a2)
        eval_mask = ~df["date"].dt.month.isin(self.exclude_months)

        # ── Plot 1: 시계열 ────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                                 gridspec_kw={"height_ratios": [3, 1]},
                                 sharex=True)
        ax, ax_d = axes

        if self.cal_start and self.cal_end:
            ax.axvspan(pd.to_datetime(self.cal_start),
                       pd.to_datetime(self.cal_end),
                       color="lightyellow", alpha=0.5,
                       label=f"Cal. period ({self.cal_start} ~ {self.cal_end})",
                       zorder=0)
        _shade_winter(ax, df["date"], self.exclude_months)

        ax.plot(df["date"], df[self.rmse_target],
                color="black", lw=1.4,
                label=f"FDR ({self.rmse_target})", zorder=5)
        ax.plot(df[eval_mask]["date"], df[eval_mask]["VWC"],
                color="tomato", lw=1.1, label="CRNP VWC (eval)", zorder=4)
        ax.plot(df[~eval_mask]["date"], df[~eval_mask]["VWC"],
                color="tomato", lw=0.7, ls="--", alpha=0.4,
                label="CRNP VWC (excluded)", zorder=4)

        # ── 클러스터 이벤트 표시 (a2* 결정에 사용된 두 시점) ──────────────
        # 전제: 해당 시점에서 θ_FDR = θ_CRNP 가 성립해야 2점 역산이 유효
        events    = res.extra.get("events", [])
        a2_all_ev = np.array([e["a2"] for e in events]) if events else np.array([])
        if len(a2_all_ev):
            lo_pct = np.percentile(a2_all_ev, self.a2_low_pct)
            hi_pct = np.percentile(a2_all_ev, self.a2_high_pct)
        else:
            lo_pct = hi_pct = 0
        first_cluster, first_other = True, True
        for e in events:
            is_cluster = lo_pct <= e["a2"] <= hi_pct
            p1_dt = pd.to_datetime(e["date_p1"])
            p2_dt = pd.to_datetime(e["date_p2"])
            col   = "crimson" if is_cluster else "gray"
            alp   = 0.9       if is_cluster else 0.3
            lw_v  = 1.2       if is_cluster else 0.6
            # 수직선: Point1=파선(--), Point2=점선(:)
            ax.axvline(p1_dt, color=col, lw=lw_v, ls="--", alpha=alp, zorder=3)
            ax.axvline(p2_dt, color=col, lw=lw_v, ls=":",  alpha=alp, zorder=3)
            # FDR θ 값 점 표시
            lbl_c = f"Cluster P1/P2 (a2={e['a2']:.3f})" if first_cluster and is_cluster else ""
            lbl_o = "Other event P1/P2" if first_other and not is_cluster else ""
            ax.scatter([p1_dt, p2_dt], [e["theta1"], e["theta2"]],
                       color=col, s=30 if is_cluster else 12,
                       marker="^", alpha=alp, zorder=6,
                       label=lbl_c if is_cluster else lbl_o)
            if is_cluster:  first_cluster = False
            else:           first_other   = False

        a2_std = float(np.std(res.extra.get("a2_all", [res.a2])))
        txt = (f"a2*  = {res.a2:.4f}  (True SHP)\n"
               f"N0   = {res.N0:.1f}\n"
               f"events = {res.extra.get('n_events', '?')} "
               f"(cluster = {res.extra.get('n_cluster', '?')})\n"
               f"a2 pct = [{self.a2_low_pct}, {self.a2_high_pct}]%\n"
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
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
        ax.set_title(
            f"[{self.station_id}] SHP-Opt (Dry-down Event)  ─  FDR vs CRNP VWC\n"
            f"a2*={res.a2:.4f} (True SHP)  |  N0={res.N0:.1f}  |  "
            f"RMSE={m['RMSE']:.4f}  |  R={m['R']:.3f}",
            fontsize=11)
        ax.grid(axis="y", ls="--", alpha=0.4)

        diff = df["VWC"] - df[self.rmse_target]
        ax_d.bar(df["date"], diff,
                 color=np.where(diff.fillna(0) >= 0, "tomato", "steelblue"),
                 width=1.5, alpha=0.7)
        ax_d.axhline(0, color="black", lw=0.8)
        ax_d.set_ylabel("Residual\n(VWC - FDR)", fontsize=9)
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

        p1 = out_dir / f"{self.station_id}_shp_opt_timeseries.png"
        fig.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  📊 {p1.name}")

        # ── Plot 2: a2 분포 (최솟값 가설 시각화) ─────────────────────────
        a2_all = np.array(res.extra.get("a2_all", []))
        if len(a2_all) > 0:
            fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

            # 왼쪽: 시간에 따른 a2 추이
            ax_ts = axes2[0]
            ev_dates = [pd.to_datetime(e["date_p1"])
                        for e in self._events]
            ax_ts.scatter(ev_dates, a2_all,
                          color="steelblue", s=25, alpha=0.6, zorder=3)

            # 하위 백분위수 군집 강조
            lo = np.percentile(a2_all, self.a2_low_pct)
            hi = np.percentile(a2_all, self.a2_high_pct)
            cluster_mask = (a2_all >= lo) & (a2_all <= hi)
            ax_ts.scatter(
                [d for d, m in zip(ev_dates, cluster_mask) if m],
                a2_all[cluster_mask],
                color="tomato", s=40, zorder=4,
                label=f"Cluster [{self.a2_low_pct},{self.a2_high_pct}]%"
            )
            ax_ts.axhline(res.a2, color="tomato", lw=2,
                          label=f"True SHP a2* = {res.a2:.4f}")
            ax_ts.axhspan(lo, hi, color="tomato", alpha=0.12)
            ax_ts.set_xlabel("Event start date")
            ax_ts.set_ylabel("a2 estimate")
            ax_ts.set_title("a2 over dry-down events")
            ax_ts.legend(fontsize=8)
            ax_ts.grid(ls="--", alpha=0.4)
            ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=30, ha="right")

            # 오른쪽: 히스토그램 + 군집 영역
            ax_hist = axes2[1]
            ax_hist.hist(a2_all, bins=25, color="steelblue",
                         alpha=0.6, edgecolor="white", label="All events")
            ax_hist.hist(a2_all[cluster_mask], bins=15,
                         color="tomato", alpha=0.7, edgecolor="white",
                         label=f"Cluster ({len(a2_all[cluster_mask])} events)")
            ax_hist.axvline(res.a2, color="darkred", lw=2, ls="--",
                            label=f"a2* = {res.a2:.4f}")
            ax_hist.set_xlabel("a2 estimate")
            ax_hist.set_ylabel("Count")
            ax_hist.set_title(
                f"a2 Distribution  (n={len(a2_all)} events)\n"
                f"Minimum Hypothesis: cluster [{self.a2_low_pct}~{self.a2_high_pct}%]"
            )
            ax_hist.legend(fontsize=8)
            ax_hist.grid(ls="--", alpha=0.4)

            plt.suptitle(
                f"[{self.station_id}] SHP-Opt: True SHP Estimation\n"
                f"a2* = {res.a2:.4f}  (Minimum Hypothesis)",
                fontsize=12)
            plt.tight_layout()
            p2 = out_dir / f"{self.station_id}_shp_opt_a2_distribution.png"
            fig2.savefig(p2, dpi=150, bbox_inches="tight")
            plt.close(fig2)
            print(f"  📊 {p2.name}")

        # ── Plot 3: 산점도 ────────────────────────────────────────────────
        p_scat = out_dir / f"{self.station_id}_shp_opt_calibration_scatter.png"
        plot_scatter(res.obs, res.vwc, "shp_opt", self.station_id, m, p_scat)

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