from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Literal, TypeVar

import numpy as np

ConstraintSense = Literal["<=", ">=", "=="]
K = TypeVar("K")


def _is_scalar(value: object) -> bool:
    return isinstance(value, (Real, np.number))


def _canonical_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


class Algebraic:
    def as_expression(self) -> Expression:
        raise NotImplementedError

    def __add__(self, other: object) -> Expression:
        return self.as_expression()._add(other)

    def __radd__(self, other: object) -> Expression:
        return _to_expression(other)._add(self)

    def __sub__(self, other: object) -> Expression:
        return self.as_expression()._add(_to_expression(other)._scale(-1.0))

    def __rsub__(self, other: object) -> Expression:
        return _to_expression(other)._add(self.as_expression()._scale(-1.0))

    def __mul__(self, other: object) -> Expression:
        return self.as_expression()._mul(other)

    def __rmul__(self, other: object) -> Expression:
        return self.as_expression()._mul(other)

    def __neg__(self) -> Expression:
        return self.as_expression()._scale(-1.0)

    def __le__(self, other: object) -> ConstraintExpression:
        return ConstraintExpression(self.as_expression() - other, "<=")

    def __ge__(self, other: object) -> ConstraintExpression:
        return ConstraintExpression(self.as_expression() - other, ">=")

    def __eq__(self, other: object) -> ConstraintExpression:  # type: ignore[override]
        return ConstraintExpression(self.as_expression() - other, "==")


@dataclass(frozen=True, slots=True, eq=False)
class Expression(Algebraic):
    linear: dict[str, float] = field(default_factory=dict)
    quadratic: dict[tuple[str, str], float] = field(default_factory=dict)
    constant: float = 0.0

    def as_expression(self) -> Expression:
        return self

    @property
    def is_linear(self) -> bool:
        return not self.quadratic

    def _add(self, other: object) -> Expression:
        other_expr = _to_expression(other)
        linear = dict(self.linear)
        for name, coefficient in other_expr.linear.items():
            linear[name] = linear.get(name, 0.0) + coefficient

        quadratic = dict(self.quadratic)
        for pair, coefficient in other_expr.quadratic.items():
            quadratic[pair] = quadratic.get(pair, 0.0) + coefficient

        return Expression(
            linear=_drop_zero_terms(linear),
            quadratic=_drop_zero_terms(quadratic),
            constant=self.constant + other_expr.constant,
        )

    def _scale(self, scalar: float) -> Expression:
        return Expression(
            linear={name: coefficient * scalar for name, coefficient in self.linear.items()},
            quadratic={
                pair: coefficient * scalar
                for pair, coefficient in self.quadratic.items()
            },
            constant=self.constant * scalar,
        )

    def _mul(self, other: object) -> Expression:
        if _is_scalar(other):
            return self._scale(float(other))

        other_expr = _to_expression(other)
        if self.quadratic or other_expr.quadratic:
            raise ValueError("Only products of linear expressions are supported")

        result = Expression(constant=self.constant * other_expr.constant)

        linear: dict[str, float] = {}
        for name, coefficient in self.linear.items():
            linear[name] = linear.get(name, 0.0) + coefficient * other_expr.constant
        for name, coefficient in other_expr.linear.items():
            linear[name] = linear.get(name, 0.0) + coefficient * self.constant

        quadratic: dict[tuple[str, str], float] = {}
        for left, left_coeff in self.linear.items():
            for right, right_coeff in other_expr.linear.items():
                pair = _canonical_pair(left, right)
                quadratic[pair] = quadratic.get(pair, 0.0) + left_coeff * right_coeff

        return Expression(
            linear=_drop_zero_terms(linear),
            quadratic=_drop_zero_terms(quadratic),
            constant=result.constant,
        )

    def evaluate(self, values: dict[str, float]) -> float:
        total = self.constant
        for name, coefficient in self.linear.items():
            total += coefficient * values[name]
        for (left, right), coefficient in self.quadratic.items():
            total += coefficient * values[left] * values[right]
        return float(total)

    @property
    def variable_names(self) -> set[str]:
        names = set(self.linear)
        for left, right in self.quadratic:
            names.add(left)
            names.add(right)
        return names


@dataclass(frozen=True, slots=True, eq=False)
class Var(Algebraic):
    name: str
    index: int | None = None

    @property
    def symbol(self) -> str:
        return self.name if self.index is None else f"{self.name}_{self.index}"

    def as_expression(self) -> Expression:
        return Expression(linear={self.symbol: 1.0})


@dataclass(frozen=True, slots=True)
class VarArray:
    name: str
    variables: tuple[Var, ...]

    def __len__(self) -> int:
        return len(self.variables)

    def __iter__(self):
        return iter(self.variables)

    def __getitem__(self, index: int) -> Var:
        return self.variables[index]

    @property
    def symbols(self) -> list[str]:
        return [variable.symbol for variable in self.variables]


@dataclass(frozen=True, slots=True, eq=False)
class ExpressionArray:
    expressions: tuple[Expression, ...]

    def __len__(self) -> int:
        return len(self.expressions)

    def __iter__(self):
        return iter(self.expressions)

    def __getitem__(self, index: int) -> Expression:
        return self.expressions[index]

    def __add__(self, other: object) -> ExpressionArray:
        return self._combine(other, lambda left, right: left + right)

    def __radd__(self, other: object) -> ExpressionArray:
        return self._combine(other, lambda left, right: right + left)

    def __sub__(self, other: object) -> ExpressionArray:
        return self._combine(other, lambda left, right: left - right)

    def __rsub__(self, other: object) -> ExpressionArray:
        return self._combine(other, lambda left, right: right - left)

    def __mul__(self, other: object) -> ExpressionArray:
        if not _is_scalar(other):
            raise TypeError("ExpressionArray only supports scalar multiplication")
        return ExpressionArray(tuple(expr * other for expr in self))

    def __rmul__(self, other: object) -> ExpressionArray:
        return self.__mul__(other)

    def __neg__(self) -> ExpressionArray:
        return ExpressionArray(tuple(-expr for expr in self))

    def __le__(self, other: object) -> list[ConstraintExpression]:
        return self._compare(other, "<=")

    def __ge__(self, other: object) -> list[ConstraintExpression]:
        return self._compare(other, ">=")

    def __eq__(self, other: object) -> list[ConstraintExpression]:  # type: ignore[override]
        return self._compare(other, "==")

    def _combine(self, other: object, op) -> ExpressionArray:
        other_values = _coerce_array_operand(other, len(self))
        return ExpressionArray(
            tuple(op(left, right) for left, right in zip(self, other_values, strict=True))
        )

    def _compare(
        self, other: object, sense: ConstraintSense
    ) -> list[ConstraintExpression]:
        other_values = _coerce_array_operand(other, len(self))
        return [
            ConstraintExpression(left - right, sense)
            for left, right in zip(self, other_values, strict=True)
        ]


@dataclass(frozen=True, slots=True)
class ConstraintExpression:
    expression: Expression
    sense: ConstraintSense

    def __bool__(self) -> bool:
        raise TypeError(
            "ConstraintExpression cannot be used as a boolean; split chained "
            "constraints into separate expressions"
        )

    def as_linear_row(self, variable_names: list[str]) -> tuple[np.ndarray, float]:
        if not self.expression.is_linear:
            raise ValueError("Constraint is not linear")
        row = np.zeros(len(variable_names), dtype=float)
        positions = {name: index for index, name in enumerate(variable_names)}
        for name, coefficient in self.expression.linear.items():
            if name not in positions:
                raise ValueError(f"Unknown variable in constraint: {name}")
            row[positions[name]] = coefficient
        rhs = -float(self.expression.constant)
        return row, rhs


def as_expression(value: object) -> Expression:
    return _to_expression(value)


def sum_(items: Iterable[object]) -> Expression:
    result = Expression()
    for item in items:
        result += item
    return result


def dot(coefficients: Sequence[float], variables: VarArray) -> Expression:
    if len(coefficients) != len(variables):
        raise ValueError("Coefficient and variable lengths do not match")
    result = Expression()
    for coefficient, variable in zip(coefficients, variables, strict=True):
        result += float(coefficient) * variable
    return result


def quad_form(variables: VarArray, matrix: np.ndarray | Sequence[Sequence[float]]) -> Expression:
    matrix_array = np.asarray(matrix, dtype=float)
    if matrix_array.shape != (len(variables), len(variables)):
        raise ValueError(
            "Quadratic matrix shape mismatch: "
            f"expected {(len(variables), len(variables))}, got {matrix_array.shape}"
        )
    result = Expression()
    for row in range(len(variables)):
        for col in range(len(variables)):
            coefficient = matrix_array[row, col]
            if coefficient != 0:
                result += float(coefficient) * variables[row] * variables[col]
    return result


def matmul(matrix: np.ndarray | Sequence[Sequence[float]], variables: VarArray) -> ExpressionArray:
    matrix_array = np.asarray(matrix, dtype=float)
    if matrix_array.ndim != 2 or matrix_array.shape[1] != len(variables):
        raise ValueError(
            "Matrix and variable length mismatch: "
            f"matrix shape {matrix_array.shape}, variables {len(variables)}"
        )
    return ExpressionArray(tuple(dot(row, variables) for row in matrix_array))


def mean(value: ExpressionArray | Iterable[object]) -> Expression:
    expressions = list(value)
    if not expressions:
        raise ValueError("Cannot take mean of an empty expression list")
    return sum_(expressions) * (1.0 / len(expressions))


def square(value: Expression | Var | ExpressionArray) -> Expression | ExpressionArray:
    if isinstance(value, ExpressionArray):
        return ExpressionArray(tuple(expr * expr for expr in value))
    expression = _to_expression(value)
    return expression * expression


def _to_expression(value: object) -> Expression:
    if isinstance(value, Expression):
        return value
    if isinstance(value, Var):
        return value.as_expression()
    if _is_scalar(value):
        return Expression(constant=float(value))
    raise TypeError(f"Cannot convert {type(value).__name__} to Expression")


def _coerce_array_operand(value: object, length: int) -> tuple[object, ...]:
    if isinstance(value, ExpressionArray):
        if len(value) != length:
            raise ValueError("Expression arrays must have the same length")
        return tuple(value)
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.shape[0] != length:
            raise ValueError(
                "Expression array and numpy array length mismatch: "
                f"expected {length}, got shape {value.shape}"
            )
        return tuple(value.tolist())
    if isinstance(value, Sequence) and not isinstance(value, str):
        if len(value) != length:
            raise ValueError(
                "Expression array and sequence length mismatch: "
                f"expected {length}, got {len(value)}"
            )
        return tuple(value)
    if not isinstance(value, str):
        try:
            array_value = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            array_value = None
        if array_value is not None and array_value.ndim == 1:
            if array_value.shape[0] != length:
                raise ValueError(
                    "Expression array and array-like length mismatch: "
                    f"expected {length}, got shape {array_value.shape}"
                )
            return tuple(array_value.tolist())
    return tuple(value for _ in range(length))


def _drop_zero_terms(terms: dict[K, float]) -> dict[K, float]:
    return {key: value for key, value in terms.items() if abs(value) > 1.0e-12}


Expr = Expression
ConstraintExpr = ConstraintExpression
