import numpy as np
import pandas as pd


def normalize(x: np.ndarray, denominator="max"):
    if denominator == "max":
        denominator = np.max(x)
        if denominator == 0:
            return x, 0.0
        return x / denominator, denominator
    else:
        raise NotImplementedError


def divide_by_volatility(time_series: pd.DataFrame, split_points: list):
    log_returns = np.log(time_series / time_series.shift(1))
    volatilities = log_returns.std() * np.sqrt(252)
    volatilities = volatilities.dropna()

    thresholds = volatilities.quantile(split_points).values
    bins = [-np.inf] + list(thresholds) + [np.inf]
    binned = pd.cut(volatilities, bins=bins, labels=False, include_lowest=True)

    groups = []
    for i in range(len(bins) - 1):
        groups.append(volatilities.index[binned == i].tolist())

    return groups
