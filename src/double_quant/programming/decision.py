from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Self

import numpy as np

from double_quant.algorithm.grover import build_sfs_grover_circuit
from double_quant.algorithm.rasengan import LinearConstraintBinaryProblem
from double_quant.common import LinearSystem, QUBOProblem
from double_quant.programming.base import FinancialProgram
from double_quant.programming.expression import (
    ConstraintExpression,
    Expression,
    Var,
    VarArray,
    as_expression,
)

VariableType = Literal["continuous", "binary", "integer"]
DecisionSense = Literal["find", "minimize", "maximize"]


@dataclass(frozen=True, slots=True)
class VariableSpec:
    name: str
    length: int
    vtype: VariableType = "continuous"
    lb: float | int | None = None
    ub: float | int | None = None
    symbols: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class DecisionProgram(FinancialProgram):
    """Finance-level decision model with structured variables and constraints."""

    variables: dict[str, VariableSpec] = field(default_factory=dict)
    constraints: list[ConstraintExpression] = field(default_factory=list)
    objective: Expression | None = None
    sense: DecisionSense = "find"

    def __post_init__(self) -> None:
        if self.kind != "decision":
            raise ValueError("DecisionProgram kind must be 'decision'")

    def add_variables(
        self,
        name: str,
        length: int,
        *,
        vtype: VariableType = "continuous",
        lb: float | int | None = None,
        ub: float | int | None = None,
    ) -> VarArray:
        if length <= 0:
            raise ValueError("length must be positive")
        if name in self.variables:
            raise ValueError(f"Variable group already exists: {name}")
        variables = tuple(Var(name, index) for index in range(length))
        symbols = tuple(variable.symbol for variable in variables)
        existing = set(self.variable_names)
        duplicates = existing.intersection(symbols)
        if duplicates:
            duplicate_list = ", ".join(sorted(duplicates))
            raise ValueError(f"Variable symbols already exist: {duplicate_list}")

        self.variables[name] = VariableSpec(
            name=name,
            length=length,
            vtype=vtype,
            lb=lb,
            ub=ub,
            symbols=symbols,
        )
        return VarArray(name=name, variables=variables)

    def add_constraint(self, expression: ConstraintExpression) -> Self:
        if not isinstance(expression, ConstraintExpression):
            raise TypeError("add_constraint expects a ConstraintExpression")
        self._validate_expression_variables(expression.expression)
        self.constraints.append(expression)
        return self

    def add_constraints(self, expressions: list[ConstraintExpression]) -> Self:
        for expression in expressions:
            self.add_constraint(expression)
        return self

    def set_objective(
        self,
        expression: Expression | Var | float | int | None,
        *,
        sense: DecisionSense = "find",
    ) -> Self:
        if expression is None:
            if sense != "find":
                raise ValueError("A non-find sense requires an objective expression")
            self.objective = None
            self.sense = sense
            return self

        objective = as_expression(expression)
        self._validate_expression_variables(objective)
        self.objective = objective
        self.sense = sense
        return self

    @property
    def variable_names(self) -> list[str]:
        names: list[str] = []
        for spec in self.variables.values():
            names.extend(spec.symbols)
        return names

    def to_qubo_problem(self) -> QUBOProblem:
        """Convert an unconstrained binary quadratic model to QUBOProblem."""

        if self.objective is None:
            raise ValueError("QUBO conversion requires an objective")
        if self.constraints:
            raise ValueError("QUBO conversion currently supports unconstrained models")
        non_binary = [
            spec.name for spec in self.variables.values() if spec.vtype != "binary"
        ]
        if non_binary:
            raise ValueError(
                "QUBO conversion requires binary variables; non-binary groups: "
                + ", ".join(non_binary)
            )
        objective = self._objective_for_minimization()
        variable_names = self.variable_names
        positions = {name: index for index, name in enumerate(variable_names)}
        matrix = np.zeros((len(variable_names), len(variable_names)), dtype=float)

        for name, coefficient in objective.linear.items():
            matrix[positions[name], positions[name]] += coefficient
        for (left, right), coefficient in objective.quadratic.items():
            row = positions[left]
            col = positions[right]
            matrix[row, col] += coefficient

        return QUBOProblem(
            matrix,
            constant=objective.constant,
            variable_names=variable_names,
        )

    def to_linear_system(self) -> LinearSystem:
        """Extract Ax=b from linear equality constraints for HHL-style solvers."""

        if not self.constraints:
            raise ValueError("Linear-system conversion requires constraints")
        if any(constraint.sense != "==" for constraint in self.constraints):
            raise ValueError("Linear-system conversion only supports equality constraints")

        variable_names = self.variable_names
        rows: list[np.ndarray] = []
        rhs: list[float] = []
        for constraint in self.constraints:
            row, value = constraint.as_linear_row(variable_names)
            rows.append(row)
            rhs.append(value)

        matrix = np.vstack(rows).astype(float)
        vector = np.asarray(rhs, dtype=float)
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                "Linear-system conversion requires a square system: "
                f"got {matrix.shape[0]} constraints for {matrix.shape[1]} variables"
            )
        return LinearSystem(matrix, vector)

    def to_rasengan_problem(
        self,
        *,
        penalty: float = 400.0,
    ) -> LinearConstraintBinaryProblem:
        """Convert a binary equality-constrained decision model to Rasengan input."""

        non_binary = [
            spec.name for spec in self.variables.values() if spec.vtype != "binary"
        ]
        if non_binary:
            raise ValueError(
                "Rasengan conversion requires binary variables; non-binary groups: "
                + ", ".join(non_binary)
            )
        if any(constraint.sense != "==" for constraint in self.constraints):
            raise ValueError("Rasengan conversion only supports equality constraints")

        variable_names = self.variable_names
        objective = self.objective or Expression()
        self._validate_expression_variables(objective)
        linear = np.zeros(len(variable_names), dtype=float)
        quadratic = np.zeros((len(variable_names), len(variable_names)), dtype=float)
        positions = {name: index for index, name in enumerate(variable_names)}

        for name, coefficient in objective.linear.items():
            linear[positions[name]] += coefficient
        for (left, right), coefficient in objective.quadratic.items():
            quadratic[positions[left], positions[right]] += coefficient

        if self.constraints:
            rows: list[np.ndarray] = []
            rhs: list[float] = []
            for constraint in self.constraints:
                row, value = constraint.as_linear_row(variable_names)
                rows.append(row)
                rhs.append(value)
            constraints = np.vstack(rows).astype(float)
            rhs_array = np.asarray(rhs, dtype=float)
        else:
            constraints = np.zeros((0, len(variable_names)), dtype=float)
            rhs_array = np.zeros(0, dtype=float)

        if self.sense == "maximize":
            sense = "max"
        elif self.sense in {"minimize", "find"}:
            sense = "min"
        else:
            raise ValueError("Rasengan conversion requires sense='minimize' or 'maximize'")

        return LinearConstraintBinaryProblem(
            linear=linear,
            constraints=constraints,
            rhs=rhs_array,
            sense=sense,
            quadratic=quadratic if np.any(quadratic) else None,
            penalty=penalty,
            variable_names=tuple(variable_names),
        )

    def to_grover_circuit(
        self,
        *,
        iterations: int = 1,
        compressed_qubits: int | None = None,
        marked_state: str | None = None,
    ):
        """Build an SFS-Grover search circuit over this binary decision space."""

        non_binary = [
            spec.name for spec in self.variables.values() if spec.vtype != "binary"
        ]
        if non_binary:
            raise ValueError(
                "Grover conversion requires binary variables; non-binary groups: "
                + ", ".join(non_binary)
            )
        logical_variables = len(self.variable_names)
        if logical_variables == 0:
            raise ValueError("Grover conversion requires at least one variable")
        return build_sfs_grover_circuit(
            logical_variables=logical_variables,
            iterations=iterations,
            compressed_qubits=compressed_qubits,
            marked_state=marked_state,
        )

    def _objective_for_minimization(self) -> Expression:
        if self.objective is None:
            raise ValueError("Objective is not set")
        if self.sense == "minimize":
            return self.objective
        if self.sense == "maximize":
            return self.objective * -1.0
        raise ValueError("QUBO conversion requires sense='minimize' or 'maximize'")

    def _validate_expression_variables(self, expression: Expression) -> None:
        known = set(self.variable_names)
        unknown = sorted(expression.variable_names.difference(known))
        if unknown:
            raise ValueError("Unknown variables in expression: " + ", ".join(unknown))


DecisionProblem = DecisionProgram
