from __future__ import annotations

from dataclasses import replace
from typing import Final, TypeVar

from double_quant.programming.decision import DecisionProgram
from double_quant.programming.expression import (
    ConstraintExpression,
    Expression,
    as_expression,
)

ZERO_TOLERANCE: Final = 1.0e-12
K = TypeVar("K")


def optimize_expression(expression: object, *, atol: float = ZERO_TOLERANCE) -> Expression:
    """Return a canonical expression with redundant zero terms removed."""

    expr = as_expression(expression)
    return Expression(
        linear=_drop_zero_terms(expr.linear, atol),
        quadratic=_drop_zero_terms(expr.quadratic, atol),
        constant=_drop_zero_value(expr.constant, atol),
    )


def optimize_constraint(
    constraint: ConstraintExpression,
    *,
    atol: float = ZERO_TOLERANCE,
) -> ConstraintExpression:
    """Return a constraint with its expression canonicalized."""

    return ConstraintExpression(
        optimize_expression(constraint.expression, atol=atol),
        constraint.sense,
    )


def optimize_decision_program(
    program: DecisionProgram,
    *,
    inplace: bool = False,
    atol: float = ZERO_TOLERANCE,
) -> DecisionProgram:
    """Canonicalize a decision program as a standalone compiler-style pass."""

    optimized = program if inplace else _copy_decision_program(program)
    if optimized.objective is not None:
        optimized.objective = optimize_expression(optimized.objective, atol=atol)
    optimized.constraints = [
        optimize_constraint(constraint, atol=atol) for constraint in optimized.constraints
    ]
    return optimized


def _copy_decision_program(program: DecisionProgram) -> DecisionProgram:
    return replace(
        program,
        data=dict(program.data),
        parameters=dict(program.parameters),
        assumptions=list(program.assumptions),
        outputs=list(program.outputs),
        variables=dict(program.variables),
        constraints=list(program.constraints),
    )


def _drop_zero_terms(terms: dict[K, float], atol: float) -> dict[K, float]:
    return {key: value for key, value in terms.items() if abs(value) > atol}


def _drop_zero_value(value: float, atol: float) -> float:
    return 0.0 if abs(value) <= atol else float(value)
