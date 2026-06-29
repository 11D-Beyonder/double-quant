from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit

from double_quant.algorithm.rasengan import (
    LinearConstraintBinaryProblem,
    build_penalty_qaoa_circuit,
    build_rasengan_circuit,
)
from double_quant.application._rasengan_factories import (
    RasenganProblemInstance,
    payment_settlement_instance,
)


@dataclass(frozen=True, slots=True)
class PaymentSettlementAlgorithm:
    """Func-7 payment and settlement liquidity-neutral search."""

    accounts: int = 3
    layers: int = 1

    def build_rasengan_instance(self) -> RasenganProblemInstance:
        return payment_settlement_instance(pair_count=self.accounts)

    def build_problem(self) -> LinearConstraintBinaryProblem:
        return self.build_rasengan_instance().problem

    def build_circuit(self) -> QuantumCircuit:
        instance = self.build_rasengan_instance()
        circuit = build_rasengan_circuit(
            instance.problem,
            layers=self.layers,
            transition_basis=instance.transition_basis,
            feasible_state=instance.feasible_state,
        )
        circuit.metadata = {**(circuit.metadata or {}), "application_id": "Func-7"}
        return circuit

    def build_baseline_circuit(self) -> QuantumCircuit:
        circuit = build_penalty_qaoa_circuit(
            self.build_rasengan_instance().problem,
            layers=self.layers,
        )
        circuit.metadata = {**(circuit.metadata or {}), "application_id": "Func-7"}
        return circuit
