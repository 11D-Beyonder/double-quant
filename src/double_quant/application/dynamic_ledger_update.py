from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit

from double_quant.algorithm.shor import build_shor_period_finding_circuit
from double_quant.algorithm.shor.baseline import (
    build_unoptimized_shor_period_finding_circuit,
)


@dataclass(frozen=True, slots=True)
class DynamicLedgerUpdateAlgorithm:
    """Func-4 dynamic ledger update modeled as Shor period finding."""

    modulus: int = 15
    base: int = 2
    phase_qubits: int | None = None

    def build_circuit(self) -> QuantumCircuit:
        circuit = build_shor_period_finding_circuit(
            self.modulus,
            base=self.base,
            phase_qubits=self.phase_qubits,
        )
        circuit.metadata = {**(circuit.metadata or {}), "application_id": "Func-4"}
        return circuit

    def build_baseline_circuit(self) -> QuantumCircuit:
        circuit = build_unoptimized_shor_period_finding_circuit(
            self.modulus,
            base=self.base,
            phase_qubits=self.phase_qubits,
        )
        circuit.metadata = {**(circuit.metadata or {}), "application_id": "Func-4"}
        return circuit
