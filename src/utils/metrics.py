# src/utils/metrics.py
"""교정 성능 지표 계산"""

import numpy as np


def compute_metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    """
    관측값(obs)과 예측값(pred)으로 성능 지표 계산.
    NaN은 자동으로 제외.
    """
    valid = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[valid], pred[valid]
    n = int(len(o))

    if n < 2:
        return dict(RMSE=np.nan, MAE=np.nan, Bias=np.nan,
                    R=np.nan, NSE=np.nan, n=n)

    rmse = float(np.sqrt(np.mean((o - p) ** 2)))
    mae  = float(np.mean(np.abs(o - p)))
    bias = float(np.mean(p - o))
    r    = float(np.corrcoef(o, p)[0, 1])

    # Nash-Sutcliffe Efficiency
    ss_res = np.sum((o - p) ** 2)
    ss_tot = np.sum((o - np.mean(o)) ** 2)
    nse = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return dict(RMSE=rmse, MAE=mae, Bias=bias, R=r, NSE=nse, n=n)