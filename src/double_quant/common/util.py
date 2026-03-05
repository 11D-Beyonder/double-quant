import numpy as np
import pandas as pd
import rich


def normalize(x: np.ndarray, denominator="max"):
    if denominator == "max":
        denominator = np.max(x)
        if denominator == 0:
            return x, 0.0
        return x / denominator, denominator
    else:
        raise NotImplementedError


def divide_by_volatility(returns: pd.DataFrame, split_points: list):
    """
    Divide assets into risk buckets based on annualized volatility quantile.

    Args:
        time_series: DataFrame of asset prices or returns. If prices, it converts to returns.
        split_points: List of quantiles to split the data. e.g. [0.3, 0.7] divides into Low (bottom 30%), Mid (30-70%), High (top 30%).

    Returns:
        List of lists, where each inner list contains asset names (columns) for a bucket.
        Buckets are ordered from Low Volatility to High Volatility.
    """
    volatilities = returns.std() * np.sqrt(252)
    volatilities = volatilities.dropna()

    thresholds = volatilities.quantile(split_points).values
    # bins: [-inf, q_0.3, q_0.7, inf]
    bins = [-np.inf] + list(thresholds) + [np.inf]
    binned = pd.cut(
        volatilities, bins=bins, labels=False, include_lowest=True, duplicates="drop"
    )

    groups = []
    # len(bins) - 1 is the number of groups
    for i in range(len(bins) - 1):
        groups.append(volatilities.index[binned == i].tolist())

    return groups


def warning(msg: str):
    rich.print(f":warning:{msg}")
