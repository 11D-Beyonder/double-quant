from __future__ import annotations

import math

from qiskit import QuantumCircuit

from double_quant.algorithm.shor.circuit import build_generic_shor_period_finding_circuit


def build_unoptimized_shor_period_finding_circuit(
    modulus: int,
    *,
    base: int = 2,
    phase_qubits: int | None = None,
    work_qubits: int | None = None,
) -> QuantumCircuit:
    """Build the quantum Shor baseline with generic modular-permutation gates."""
    circuit = build_generic_shor_period_finding_circuit(
        modulus,
        base=base,
        phase_qubits=phase_qubits,
        work_qubits=work_qubits,
    )
    circuit.name = f"shor_baseline_N{modulus}_a{base}"
    circuit.metadata = {
        **(circuit.metadata or {}),
        "variant": "generic_permutation_baseline",
        "baseline_type": "quantum",
    }
    return circuit


def classical_trial_division_operations(modulus: int) -> int:
    """Return the trial divisions required by the simple classical baseline."""
    if modulus <= 1:
        raise ValueError("modulus must be greater than 1")
    return max(1, math.isqrt(modulus) - 1)
