from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np
from numpy.typing import NDArray


BinarySense = Literal["min", "max"]


@dataclass(frozen=True, slots=True)
class LinearConstraintBinaryProblem:
    """Linear constrained binary optimization model used by Rasengan circuits."""

    linear: NDArray[np.float64]
    constraints: NDArray[np.float64]
    rhs: NDArray[np.float64]
    sense: BinarySense = "min"
    quadratic: NDArray[np.float64] | None = None
    penalty: float = 400.0
    variable_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        linear = np.asarray(self.linear, dtype=float)
        constraints = np.asarray(self.constraints, dtype=float)
        rhs = np.asarray(self.rhs, dtype=float)
        quadratic = None if self.quadratic is None else np.asarray(self.quadratic, dtype=float)

        if linear.ndim != 1:
            raise ValueError("linear must be a one-dimensional array")
        if constraints.ndim != 2:
            raise ValueError("constraints must be a two-dimensional array")
        if constraints.shape[1] != linear.shape[0]:
            raise ValueError("constraints column count must match variable count")
        if rhs.shape != (constraints.shape[0],):
            raise ValueError("rhs length must match constraint count")
        if self.sense not in {"min", "max"}:
            raise ValueError("sense must be 'min' or 'max'")
        if quadratic is not None and quadratic.shape != (linear.shape[0], linear.shape[0]):
            raise ValueError("quadratic must be an n x n matrix")
        if self.variable_names is not None and len(self.variable_names) != linear.shape[0]:
            raise ValueError("variable_names length must match variable count")

        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "rhs", rhs)
        object.__setattr__(self, "quadratic", quadratic)
        if self.variable_names is None:
            names = tuple(f"x_{index}" for index in range(linear.shape[0]))
            object.__setattr__(self, "variable_names", names)

    @property
    def num_variables(self) -> int:
        return int(self.linear.shape[0])

    @property
    def num_constraints(self) -> int:
        return int(self.constraints.shape[0])

    @property
    def objective_direction(self) -> int:
        return 1 if self.sense == "min" else -1

    def objective_value(self, bitstring: NDArray[np.int_] | list[int]) -> float:
        bits = self._as_bits(bitstring)
        value = float(self.linear @ bits)
        if self.quadratic is not None:
            value += float(bits @ self.quadratic @ bits)
        return value

    def penalized_value(self, bitstring: NDArray[np.int_] | list[int]) -> float:
        bits = self._as_bits(bitstring)
        residual = self.constraints @ bits - self.rhs
        value = self.objective_direction * self.objective_value(bits)
        return float(value + self.penalty * (residual @ residual))

    def is_feasible(self, bitstring: NDArray[np.int_] | list[int], *, atol: float = 1e-9) -> bool:
        bits = self._as_bits(bitstring)
        return bool(np.allclose(self.constraints @ bits, self.rhs, atol=atol))

    def feasible_state(self) -> NDArray[np.int_]:
        for bits in self.iter_binary_states():
            if self.is_feasible(bits):
                return bits
        raise ValueError("no feasible state exists for the given constraints")

    def best_feasible_state(self) -> NDArray[np.int_]:
        feasible_states = [bits for bits in self.iter_binary_states() if self.is_feasible(bits)]
        if not feasible_states:
            raise ValueError("no feasible state exists for the given constraints")
        key = self.objective_value
        return min(feasible_states, key=key) if self.sense == "min" else max(feasible_states, key=key)

    def iter_binary_states(self) -> list[NDArray[np.int_]]:
        return [
            np.asarray(bits, dtype=int)
            for bits in product((0, 1), repeat=self.num_variables)
        ]

    def _as_bits(self, bitstring: NDArray[np.int_] | list[int]) -> NDArray[np.int_]:
        bits = np.asarray(bitstring, dtype=int)
        if bits.shape != (self.num_variables,):
            raise ValueError("bitstring length must match variable count")
        if np.any((bits != 0) & (bits != 1)):
            raise ValueError("bitstring must contain only 0/1 values")
        return bits
