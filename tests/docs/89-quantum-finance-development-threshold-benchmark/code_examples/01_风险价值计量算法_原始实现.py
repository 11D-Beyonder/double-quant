# ruff: noqa: F821
returns_array = np.asarray(returns, dtype=float)
if returns_array.ndim != 1 or returns_array.size == 0:
    raise ValueError("收益率序列必须是一维非空数组")
if not 0.0 < alpha < 1.0:
    raise ValueError("置信水平必须在 0 到 1 之间")
sorted_returns = np.sort(returns_array)
tail_count = max(1, int(np.ceil((1.0 - alpha) * sorted_returns.size)))
tail_losses = sorted_returns[:tail_count]
value = -float(np.mean(tail_losses))
