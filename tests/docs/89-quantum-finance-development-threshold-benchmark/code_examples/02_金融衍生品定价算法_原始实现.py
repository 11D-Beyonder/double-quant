# ruff: noqa: F821
scenario_array = np.asarray(scenarios, dtype=float)
if scenario_array.ndim != 1 or scenario_array.size == 0:
    raise ValueError("到期价格场景必须是一维非空数组")
strike = 100.0
risk_free_rate = 0.0
maturity = 1.0
payoff = np.maximum(scenario_array - strike, 0.0)
discount = np.exp(-risk_free_rate * maturity)
value = float(discount * np.mean(payoff))
