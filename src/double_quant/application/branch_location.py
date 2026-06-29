from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit

from double_quant.algorithm.grover import build_sfs_grover_circuit
from double_quant.algorithm.grover.baseline import build_plain_grover_baseline


@dataclass(frozen=True, slots=True)
class BranchLocationAlgorithm:
    """Func-9 branch facility-location search with SFS-Grover."""

    candidate_sites: int = 8
    grover_iterations: int = 2

    def build_circuit(self) -> QuantumCircuit:
        circuit = build_sfs_grover_circuit(
            logical_variables=self.candidate_sites,
            iterations=self.grover_iterations,
        )
        circuit.metadata = {**(circuit.metadata or {}), "application_id": "Func-9"}
        return circuit

    def build_baseline_circuit(self) -> QuantumCircuit:
        circuit = build_plain_grover_baseline(
            num_qubits=self.candidate_sites,
            iterations=self.grover_iterations,
        )
        circuit.metadata = {**(circuit.metadata or {}), "application_id": "Func-9"}
        return circuit
