# ruff: noqa: F821
program = ValuationProgram(name="组合风险计量", kind="valuation", domain="风险计量")
program.add_data("portfolio_returns", returns)
program.add_parameter("alpha", alpha)
program.set_measure(ExpectedShortfallMeasure, target="组合尾部风险")
value = program.evaluate()
