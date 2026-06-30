from __future__ import annotations

from qiskit import QuantumCircuit

from double_quant.algorithm.grover.circuit import build_grover_circuit


def build_plain_grover_baseline(
    *,
    num_qubits: int,
    iterations: int,
    marked_state: str | None = None,
) -> QuantumCircuit:
    """Build the ordinary Grover quantum baseline."""
    circuit = build_grover_circuit(
        num_qubits=num_qubits,
        iterations=iterations,
        marked_state=marked_state,
    )
    circuit.metadata = {
        **(circuit.metadata or {}),
        "variant": "plain_grover_baseline",
        "baseline_type": "quantum",
    }
    return circuit


def classical_exhaustive_search_operations(search_space_size: int) -> int:
    """Return the exhaustive-search operation count for a classical baseline."""
    if search_space_size <= 0:
        raise ValueError("search_space_size must be positive")
    return search_space_size


__all__ = [
    "build_grover_circuit",
    "build_plain_grover_baseline",
    "classical_exhaustive_search_operations",
]
