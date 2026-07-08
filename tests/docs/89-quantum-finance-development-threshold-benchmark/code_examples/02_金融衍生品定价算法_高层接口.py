# ruff: noqa: F821
program = ValuationProgram(name="衍生品定价", kind="valuation", domain="衍生品定价")
program.add_data("terminal_price_scenarios", scenarios)
program.add_parameter("strike", 100.0)
program.add_parameter("risk_free_rate", 0.0)
program.add_parameter("maturity", "1Y")
program.set_measure(EuropeanCallPriceMeasure, target="衍生品价格")
value = program.evaluate()
