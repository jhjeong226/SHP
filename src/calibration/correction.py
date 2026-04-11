# src/calibration/correction.py
"""
NeutronCorrector
================
시간별 CRNP 데이터에 세 가지 보정을 적용합니다.

  fi : 입사 우주선 강도 보정  (NMDB 다운로드)
  fp : 기압 보정
  fw : 절대습도 보정

보정식:
  N_corrected = N_raw * fw / (fp * fi)

입력:  HC_CRNP_hourly.xlsx  (timestamp, N_counts, Ta, RH, Pa, abs_humidity)
출력:  동일 DataFrame + fi, fp, fw, N_corrected 컬럼 추가
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd


# ── 보정 계수 계산 ────────────────────────────────────────────────────────────

def _correction_pressure(Pa: pd.Series, Pref: float,
                         L: float = 130.0) -> pd.Series:
    """
    기압 보정계수 fp.
    fp = exp((P - Pref) / L)
    L: 흡수 평균자유행정 [hPa], 기본값 130
    """
    return np.exp((Pa - Pref) / L)


def _correction_humidity(abs_hum: pd.Series, Aref: float) -> pd.Series:
    """
    절대습도 보정계수 fw.
    fw = 1 + 0.0054 * (A - Aref)
    """
    return 1.0 + 0.0054 * (abs_hum - Aref)


def _correction_incoming(incoming: pd.Series, Iref: float) -> pd.Series:
    """입사 중성자 강도 보정계수 fi = incoming / Iref"""
    return incoming / Iref


# ── NMDB 다운로드 ─────────────────────────────────────────────────────────────

def _download_nmdb(start: pd.Timestamp, end: pd.Timestamp,
                   station: str, utc_offset: int) -> Optional[pd.DataFrame]:
    """crnpy 로 NMDB 중성자 모니터 데이터 다운로드."""
    try:
        import crnpy
        nmdb = crnpy.get_incoming_neutron_flux(
            start, end,
            station=station,
            utc_offset=utc_offset,
        )
        return nmdb
    except Exception as e:
        print(f"  ⚠️  NMDB 다운로드 실패 ({station}): {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════

class NeutronCorrector:
    """
    시간별 CRNP DataFrame 에 보정을 적용합니다.

    Parameters
    ----------
    station_cfg : YAML station 섹션
    options     : YAML processing_options 섹션
    """

    def __init__(self, station_cfg: Dict, options: Dict):
        self.station_cfg = station_cfg
        self.options     = options

        cal_cfg = station_cfg.get("calibration", {})
        self.neutron_monitor = cal_cfg.get("neutron_monitor", "MXCO")
        self.utc_offset      = cal_cfg.get("utc_offset", 9)

        corr = options.get("corrections", {})
        self.do_fi = corr.get("incoming_flux", True)
        self.do_fp = corr.get("pressure",      True)
        self.do_fw = corr.get("humidity",      True)

        self.L = 130.0   # 기압 보정 상수 [hPa]

    # ── 공개 인터페이스 ──────────────────────────────────────────────────────

    def correct(self, df: pd.DataFrame,
                Pref: Optional[float] = None,
                Aref: Optional[float] = None,
                Iref: Optional[float] = None) -> pd.DataFrame:
        """
        보정 적용.

        Parameters
        ----------
        df   : HC_CRNP_hourly.xlsx 로드 결과
        Pref : 기준 기압 (None → 전체 평균)
        Aref : 기준 절대습도 (None → 전체 평균)
        Iref : 기준 입사 강도 (None → NMDB 기간 평균)

        Returns
        -------
        df 에 fi, fp, fw, N_corrected 컬럼이 추가된 DataFrame
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # ① 결측 보간 (Ta, RH, Pa — 최대 24시간 선형)
        for col in ["Ta", "RH", "Pa", "abs_humidity"]:
            if col in df.columns:
                df[col] = df[col].interpolate(
                    method="linear", limit=24, limit_direction="both"
                )

        # ② 보정 계수 초기화
        df["fi"] = 1.0
        df["fp"] = 1.0
        df["fw"] = 1.0

        # ③ 기압 보정
        if self.do_fp and "Pa" in df.columns:
            pref = Pref if Pref is not None else float(df["Pa"].mean())
            df["fp"] = _correction_pressure(df["Pa"], pref, self.L)
            print(f"  fp 보정 적용  (Pref={pref:.2f} hPa)")

        # ④ 습도 보정
        if self.do_fw and "abs_humidity" in df.columns:
            aref = Aref if Aref is not None else float(df["abs_humidity"].mean())
            df["fw"] = _correction_humidity(df["abs_humidity"], aref)
            print(f"  fw 보정 적용  (Aref={aref:.4f} g/m³)")

        # ⑤ 입사 강도 보정
        if self.do_fi:
            nmdb = _download_nmdb(
                df["timestamp"].min(), df["timestamp"].max(),
                self.neutron_monitor, self.utc_offset,
            )
            if nmdb is not None:
                # 시간 보간
                try:
                    import crnpy
                    df["incoming"] = crnpy.interpolate_incoming_flux(
                        nmdb["timestamp"], nmdb["counts"], df["timestamp"]
                    )
                except Exception:
                    df["incoming"] = np.interp(
                        df["timestamp"].astype(np.int64),
                        pd.to_datetime(nmdb["timestamp"]).astype(np.int64),
                        nmdb["counts"],
                    )
                iref = Iref if Iref is not None else float(df["incoming"].mean())
                df["fi"] = _correction_incoming(df["incoming"], iref)
                print(f"  fi 보정 적용  (Iref={iref:.2f}, monitor={self.neutron_monitor})")
            else:
                print("  fi 보정 건너뜀 (NMDB 다운로드 실패, fi=1.0 유지)")

        # ⑥ 최종 보정 중성자
        df["N_corrected"] = df["N_counts"] * df["fw"] / (df["fp"] * df["fi"])

        # ⑦ 요약
        raw_mean  = df["N_counts"].mean()
        corr_mean = df["N_corrected"].mean()
        print(f"  N_counts: {raw_mean:.1f} → N_corrected: {corr_mean:.1f} "
              f"(Δ{corr_mean - raw_mean:+.1f})")

        return df

    def reference_values(self, df: pd.DataFrame,
                         start: Optional[str] = None,
                         end:   Optional[str] = None) -> Dict[str, float]:
        """교정 기간의 참조값(Pref, Aref, Iref) 계산."""
        if start and end:
            mask = (df["timestamp"] >= pd.to_datetime(start)) & \
                   (df["timestamp"] <= pd.to_datetime(end))
            sub  = df[mask]
        else:
            sub = df

        refs: Dict[str, float] = {}
        if "Pa"           in sub: refs["Pref"] = float(sub["Pa"].mean())
        if "abs_humidity" in sub: refs["Aref"] = float(sub["abs_humidity"].mean())
        if "incoming"     in sub: refs["Iref"] = float(sub["incoming"].mean())
        return refs