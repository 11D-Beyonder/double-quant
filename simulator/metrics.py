from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np


def fidelity(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return pure-state fidelity |<reference|candidate>|^2."""

    ref = _normalized_state(reference)
    cand = _normalized_state(candidate)
    if ref.shape != cand.shape:
        raise ValueError(f"statevector shape mismatch: {ref.shape} != {cand.shape}")
    return float(abs(np.vdot(ref, cand)) ** 2)


def distribution_total_variation(
    reference: Mapping[str, float], candidate: Mapping[str, float]
) -> float:
    """Return total variation distance between two bitstring distributions."""

    ref = _normalize_distribution(reference)
    cand = _normalize_distribution(candidate)
    keys = set(ref) | set(cand)
    return float(0.5 * sum(abs(ref.get(key, 0.0) - cand.get(key, 0.0)) for key in keys))


def solution_success_probability(
    distribution: Mapping[str, float] | Mapping[str, int],
    optimal_bitstrings: set[str] | list[str] | tuple[str, ...],
) -> float:
    """Return probability mass assigned to known optimal bitstrings."""

    normalized = _normalize_distribution(distribution)
    targets = {str(bitstring) for bitstring in optimal_bitstrings}
    return float(sum(normalized.get(bitstring, 0.0) for bitstring in targets))


def bitstring_objective_accuracy(
    measured_distribution: Mapping[str, float] | Mapping[str, int],
    objective: Callable[[str], float],
    optimum_value: float,
) -> float:
    """Return expected absolute objective error under the measured distribution."""

    normalized = _normalize_distribution(measured_distribution)
    return float(
        sum(
            probability * abs(float(objective(bitstring)) - float(optimum_value))
            for bitstring, probability in normalized.items()
        )
    )


def _normalized_state(state: np.ndarray) -> np.ndarray:
    vector = np.asarray(state, dtype=complex)
    if vector.ndim != 1:
        raise ValueError("statevector must be 1-dimensional")
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("statevector norm cannot be zero")
    return vector / norm


def _normalize_distribution(
    distribution: Mapping[str, float] | Mapping[str, int],
) -> dict[str, float]:
    normalized = {str(key): float(value) for key, value in distribution.items()}
    if any(value < 0.0 for value in normalized.values()):
        raise ValueError("distribution cannot contain negative values")
    total = sum(normalized.values())
    if total <= 0.0:
        raise ValueError("distribution must contain positive mass")
    return {key: value / total for key, value in normalized.items()}
