from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

from double_quant.programming.base import FinancialProgram
from double_quant.programming.measures import MeasureFunction


@dataclass
class ValuationProgram(FinancialProgram):
    """Finance-level valuation model dispatched by an importable measure class."""

    target: str = ""
    measure: type[MeasureFunction] | None = None
    breakdown: str | None = None

    def __post_init__(self) -> None:
        if self.kind != "valuation":
            raise ValueError("ValuationProgram kind must be 'valuation'")

    def set_measure(
        self,
        measure: type[MeasureFunction],
        *,
        target: str,
        breakdown: str | None = None,
    ) -> Self:
        if not issubclass(measure, MeasureFunction):
            raise TypeError("measure must be a MeasureFunction subclass")
        self.measure = measure
        self.target = target
        self.breakdown = breakdown
        return self

    def evaluate(self) -> Any:
        if self.measure is None:
            raise ValueError("ValuationProgram measure is not set")
        missing_data = set(self.measure.required_data()).difference(self.data)
        if missing_data:
            raise ValueError("Missing measure data: " + ", ".join(sorted(missing_data)))
        missing_parameters = set(self.measure.required_parameters()).difference(
            self.parameters
        )
        if missing_parameters:
            raise ValueError(
                "Missing measure parameters: " + ", ".join(sorted(missing_parameters))
        )
        return self.measure.evaluate(self.data, self.parameters)
