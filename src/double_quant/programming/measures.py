from __future__ import annotations

from typing import Any, ClassVar, Mapping

import numpy as np

from double_quant.algorithm.shapley import BinaryEnumerationCalculator
from double_quant.application.risk import RiskAttributor
from double_quant.common.metric import expected_shortfall


class MeasureFunction:
    """Base class for importable valuation measures."""

    name: ClassVar[str] = "measure"

    @classmethod
    def required_data(cls) -> tuple[str, ...]:
        return ()

    @classmethod
    def required_parameters(cls) -> tuple[str, ...]:
        return ()

    @classmethod
    def evaluate(
        cls, data: Mapping[str, Any], parameters: Mapping[str, Any]
    ) -> Any:
        raise NotImplementedError


class ExpectedShortfallMeasure(MeasureFunction):
    name = "expected_shortfall"

    @classmethod
    def required_data(cls) -> tuple[str, ...]:
        return ("portfolio_returns",)

    @classmethod
    def evaluate(
        cls, data: Mapping[str, Any], parameters: Mapping[str, Any]
    ) -> float:
        returns = np.asarray(data["portfolio_returns"], dtype=float)
        alpha = float(parameters.get("alpha", 0.95))
        return expected_shortfall(returns, alpha)


class ShapleyRiskContributionMeasure(MeasureFunction):
    name = "shapley_risk_contribution"

    @classmethod
    def required_data(cls) -> tuple[str, ...]:
        return ("asset_returns",)

    @classmethod
    def evaluate(
        cls, data: Mapping[str, Any], parameters: Mapping[str, Any]
    ) -> dict[str, float]:
        returns = data["asset_returns"]
        alpha = float(parameters.get("alpha", 0.95))
        mode = parameters.get("mode", "rs")
        solver_class = parameters.get("solver_class", BinaryEnumerationCalculator)
        solver_kwargs = dict(parameters.get("solver_kwargs", {}))
        return RiskAttributor(
            returns,
            solver_class,
            alpha=alpha,
            mode=mode,
            **solver_kwargs,
        ).attribute()


class EuropeanCallPriceMeasure(MeasureFunction):
    name = "european_call_price"

    @classmethod
    def required_data(cls) -> tuple[str, ...]:
        return ("terminal_price_scenarios",)

    @classmethod
    def required_parameters(cls) -> tuple[str, ...]:
        return ("strike",)

    @classmethod
    def evaluate(
        cls, data: Mapping[str, Any], parameters: Mapping[str, Any]
    ) -> float:
        terminal_prices = np.asarray(data["terminal_price_scenarios"], dtype=float)
        strike = float(parameters["strike"])
        risk_free_rate = float(parameters.get("risk_free_rate", 0.0))
        maturity = _parse_year_fraction(parameters.get("maturity", 1.0))
        payoff = np.maximum(terminal_prices - strike, 0.0)
        discount = np.exp(-risk_free_rate * maturity)
        return float(discount * np.mean(payoff))


def _parse_year_fraction(value: Any) -> float:
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().upper()
        if stripped.endswith("Y"):
            return float(stripped[:-1])
        if stripped.endswith("M"):
            return float(stripped[:-1]) / 12.0
        if stripped.endswith("D"):
            return float(stripped[:-1]) / 365.0
    raise ValueError(f"Unsupported maturity value: {value!r}")
