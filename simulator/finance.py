from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit


def build_weighted_sum_circuit(
    num_assets: int,
    *,
    weights: np.ndarray | list[float] | None = None,
    entangle: bool = True,
) -> QuantumCircuit:
    """Build a shallow quantum-finance circuit encoding asset weights as rotations."""

    if num_assets < 1:
        raise ValueError("num_assets must be positive")
    if weights is None:
        values = np.linspace(0.2, 1.0, num_assets)
    else:
        values = np.asarray(weights, dtype=float)
        if values.shape != (num_assets,):
            raise ValueError(f"weights must have shape ({num_assets},)")
    max_abs = float(np.max(np.abs(values))) or 1.0

    circuit = QuantumCircuit(num_assets, name="weighted_sum_finance")
    for index, value in enumerate(values):
        theta = float(np.pi * value / max_abs)
        circuit.ry(theta, index)
    if entangle and num_assets > 1:
        for index in range(num_assets - 1):
            circuit.cx(index, index + 1)
    return circuit


def build_portfolio_qaoa_ansatz(
    expected_returns: np.ndarray | list[float],
    covariance: np.ndarray | list[list[float]],
    *,
    gamma: float = 0.4,
    beta: float = 0.2,
) -> QuantumCircuit:
    """Build a compact QAOA-style portfolio circuit for simulation analysis."""

    returns = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if returns.ndim != 1:
        raise ValueError("expected_returns must be 1-dimensional")
    if cov.shape != (returns.size, returns.size):
        raise ValueError("covariance must be square and match expected_returns")

    num_assets = returns.size
    circuit = QuantumCircuit(num_assets, name="portfolio_qaoa")
    circuit.h(range(num_assets))

    scale = float(max(np.max(np.abs(returns)), np.max(np.abs(cov)), 1.0))
    for index, value in enumerate(returns):
        circuit.rz(float(2.0 * gamma * value / scale), index)
    for row in range(num_assets):
        for col in range(row + 1, num_assets):
            if cov[row, col] == 0:
                continue
            angle = float(2.0 * gamma * cov[row, col] / scale)
            circuit.cx(row, col)
            circuit.rz(angle, col)
            circuit.cx(row, col)
    for index in range(num_assets):
        circuit.rx(float(2.0 * beta), index)
    return circuit
