from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit

from double_quant.algorithm.grover import build_sfs_grover_circuit
from double_quant.algorithm.grover.baseline import build_plain_grover_baseline


@dataclass(frozen=True, slots=True)
class DefiManagementAlgorithm:
    """Func-5 DeFi management as SFS-compressed Grover search."""

    logical_variables: int = 8
    grover_iterations: int = 2

    def build_circuit(self) -> QuantumCircuit:
        circuit = build_sfs_grover_circuit(
            logical_variables=self.logical_variables,
            iterations=self.grover_iterations,
        )
        circuit.metadata = {**(circuit.metadata or {}), "application_id": "Func-5"}
        return circuit

    def build_baseline_circuit(self) -> QuantumCircuit:
        circuit = build_plain_grover_baseline(
            num_qubits=self.logical_variables,
            iterations=self.grover_iterations,
        )
        circuit.metadata = {**(circuit.metadata or {}), "application_id": "Func-5"}
        return circuit
