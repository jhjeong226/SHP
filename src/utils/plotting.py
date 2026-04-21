# src/utils/plotting.py
"""공통 시각화 유틸리티"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# 방법별 고정 색상
METHOD_COLORS = {
    "standard":   "#4477AA",
    "uts":        "#CCBB44",
    "shp_joint":  "#EE6677",
    "shp_2pt":    "#228833",
    "fdr":        "#333333",
}

METHOD_LABELS = {
    "standard":   "Standard",
    "uts":        "UTS",
    "shp_joint":  "SHP-Joint",
    "shp_2pt":    "SHP-2pt",
    "fdr":        "FDR (reference)",
}


def get_color(method: str) -> str:
    return METHOD_COLORS.get(method.lower(), "#888888")


def get_label(method: str) -> str:
    return METHOD_LABELS.get(method.lower(), method)


def savefig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved: {path.name}")


def add_metrics_text(ax: plt.Axes, metrics: dict, method: str,
                     loc: str = "upper left") -> None:
    """축에 성능 지표 텍스트 박스 추가"""
    text = (f"{get_label(method)}\n"
            f"RMSE={metrics.get('RMSE', np.nan):.4f}\n"
            f"R={metrics.get('R', np.nan):.3f}\n"
            f"n={metrics.get('n', 0)}")

    locs = {
        "upper left":  (0.03, 0.97, "top"),
        "upper right": (0.97, 0.97, "top"),
        "lower left":  (0.03, 0.03, "bottom"),
    }
    x, y, va = locs.get(loc, (0.03, 0.97, "top"))
    ha = "left" if x < 0.5 else "right"

    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=8, va=va, ha=ha,
            bbox=dict(facecolor="white", edgecolor="gray",
                      alpha=0.85, boxstyle="round"))


def format_date_axis(ax: plt.Axes, rotation: int = 45) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=rotation)


def plot_scatter(obs: np.ndarray,
                 pred: np.ndarray,
                 method: str,
                 station_id: str,
                 metrics: dict,
                 out_path: Path,
                 dpi: int = 150) -> None:
    """FDR vs CRNP 산점도 생성"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))

    valid = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[valid], pred[valid]

    color = get_color(method)
    ax.scatter(o, p, color=color, alpha=0.5, s=20, edgecolors="white", lw=0.5)

    # 1:1 line
    lims = [0, 0.6]
    ax.plot(lims, lims, color="gray", ls="--", lw=1, zorder=0)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    ax.set_xlabel(r"FDR $\theta$ (m³/m³)", fontsize=11)
    ax.set_ylabel(r"CRNP VWC (m³/m³)", fontsize=11)

    title = f"[{station_id}] {get_label(method)} Scatter\nFDR vs CRNP Soil Moisture"
    ax.set_title(title, fontsize=12)

    # 통계치 텍스트
    add_metrics_text(ax, metrics, method, loc="upper left")

    ax.grid(ls="--", alpha=0.3)

    savefig(fig, out_path, dpi=dpi)


def add_rain_bars(ax_main: plt.Axes,
                  dates:   "pd.Series",
                  rain:    "pd.Series",
                  color:   str = "steelblue",
                  alpha:   float = 0.5) -> plt.Axes:
    """
    시계열 그래프 상단에 강수 막대 추가 (보조 오른쪽 Y축, 역방향).

    Parameters
    ----------
    ax_main : 기존 VWC 시계열 ax
    dates   : datetime series
    rain    : 일강수량 series (mm)
    Returns : ax_rain (오른쪽 보조축)
    """
    ax_rain = ax_main.twinx()
    valid = rain.notna() & (rain > 0)
    if valid.any():
        ax_rain.bar(dates[valid], rain[valid],
                    color=color, alpha=alpha, width=1.0,
                    zorder=2, label="Rain (mm)")
    # 역방향: 0이 위, 최대값이 아래 → 시각적으로 강수가 아래로 내리는 모양
    rain_max = max(rain.max() if rain.max() > 0 else 1, 10)
    ax_rain.set_ylim(rain_max * 4, 0)   # 역축: 상단=0, 하단=max*4 (압축)
    ax_rain.set_ylabel("Rain (mm)", fontsize=9, color=color)
    ax_rain.tick_params(axis="y", labelcolor=color, labelsize=8)
    ax_rain.yaxis.set_label_position("right")
    ax_rain.yaxis.tick_right()
    # 강수 없으면 눈금 숨김
    if not valid.any():
        ax_rain.set_yticks([])
    return ax_rain