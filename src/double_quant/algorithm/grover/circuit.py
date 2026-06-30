from __future__ import annotations

import math

from qiskit import QuantumCircuit


def build_grover_circuit(
    *,
    num_qubits: int,
    iterations: int,
    marked_state: str | None = None,
    name: str | None = None,
) -> QuantumCircuit:
    """Build a standard Grover amplitude-amplification circuit."""
    _validate_grover_inputs(num_qubits, iterations)
    marked = _resolve_marked_state(num_qubits, marked_state)

    circuit = QuantumCircuit(num_qubits, num_qubits, name=name or f"grover_n{num_qubits}")
    circuit.h(range(num_qubits))
    for _ in range(iterations):
        _append_phase_oracle(circuit, marked)
        _append_diffusion(circuit, num_qubits)
    circuit.measure(range(num_qubits), range(num_qubits)[::-1])
    circuit.metadata = {
        "algorithm": "grover",
        "num_qubits": num_qubits,
        "iterations": iterations,
        "marked_state": marked,
        "search_space_size": 2**num_qubits,
    }
    return circuit


def build_sfs_grover_circuit(
    *,
    logical_variables: int,
    iterations: int,
    compressed_qubits: int | None = None,
    marked_state: str | None = None,
) -> QuantumCircuit:
    """Build the SFS-compressed Grover circuit used by the finance applications."""
    if logical_variables <= 0:
        raise ValueError("logical_variables must be positive")
    search_qubits = compressed_qubits or max(1, math.ceil(logical_variables / 2))
    if search_qubits > logical_variables:
        raise ValueError("compressed_qubits cannot exceed logical_variables")

    circuit = build_grover_circuit(
        num_qubits=search_qubits,
        iterations=iterations,
        marked_state=marked_state,
        name=f"sfs_grover_m{logical_variables}_q{search_qubits}",
    )
    circuit.metadata = {
        **(circuit.metadata or {}),
        "algorithm": "sfs_grover",
        "logical_variables": logical_variables,
        "compressed_qubits": search_qubits,
        "search_space_size": 2**search_qubits,
        "uncompressed_search_space_size": 2**logical_variables,
    }
    return circuit


def _validate_grover_inputs(num_qubits: int, iterations: int) -> None:
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")


def _resolve_marked_state(num_qubits: int, marked_state: str | None) -> str:
    if marked_state is None:
        return "1" * num_qubits
    if len(marked_state) != num_qubits:
        raise ValueError("marked_state length must equal num_qubits")
    if set(marked_state) - {"0", "1"}:
        raise ValueError("marked_state must be a bit string")
    return marked_state


def _append_phase_oracle(circuit: QuantumCircuit, marked_state: str) -> None:
    qubits = list(range(len(marked_state)))
    for index, bit in enumerate(reversed(marked_state)):
        if bit == "0":
            circuit.x(index)
    _append_multi_controlled_z(circuit, qubits)
    for index, bit in enumerate(reversed(marked_state)):
        if bit == "0":
            circuit.x(index)


def _append_diffusion(circuit: QuantumCircuit, num_qubits: int) -> None:
    qubits = list(range(num_qubits))
    circuit.h(qubits)
    circuit.x(qubits)
    _append_multi_controlled_z(circuit, qubits)
    circuit.x(qubits)
    circuit.h(qubits)


def _append_multi_controlled_z(circuit: QuantumCircuit, qubits: list[int]) -> None:
    if len(qubits) == 1:
        circuit.z(qubits[0])
        return
    target = qubits[-1]
    controls = qubits[:-1]
    circuit.h(target)
    circuit.mcx(controls, target)
    circuit.h(target)
