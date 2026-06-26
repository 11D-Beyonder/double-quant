from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Self

ProgramKind = Literal["decision", "valuation"]


@dataclass
class FinancialProgram:
    """Base container for finance-level problem definitions."""

    name: str
    kind: ProgramKind
    domain: str
    data: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def add_data(self, name: str, value: Any) -> Self:
        self.data[name] = value
        return self

    def add_parameter(self, name: str, value: Any) -> Self:
        self.parameters[name] = value
        return self

    def add_assumption(self, assumption: str) -> Self:
        self.assumptions.append(assumption)
        return self

    def add_output(self, output: str) -> Self:
        self.outputs.append(output)
        return self


FinancialProblem = FinancialProgram
