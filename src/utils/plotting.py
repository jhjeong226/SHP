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