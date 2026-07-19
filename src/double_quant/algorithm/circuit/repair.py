from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

CircuitExecutionMode = Literal["statevector", "sampling"]


@dataclass(frozen=True, slots=True)
class RepairDiagnostic:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class RepairResult:
    circuit: QuantumCircuit
    diagnostics: tuple[RepairDiagnostic, ...]
    applied_fixes: tuple[str, ...]


class CircuitRepairError(ValueError):
    """Raised when a circuit issue is detected but cannot be repaired safely."""


class QuantumProgramRepairer:
    """Repair common circuit/runtime mismatches before execution."""

    def __init__(
        self,
        *,
        mode: CircuitExecutionMode = "statevector",
        basis_gates: Sequence[str] | None = None,
        optimization_level: int = 1,
    ) -> None:
        if mode not in {"statevector", "sampling"}:
            raise ValueError("mode must be 'statevector' or 'sampling'")
        if optimization_level not in {0, 1, 2, 3}:
            raise ValueError("optimization_level must be 0, 1, 2, or 3")
        self._mode = mode
        self._basis_gates = tuple(basis_gates) if basis_gates is not None else None
        self._optimization_level = optimization_level

    def repair(self, circuit: QuantumCircuit) -> RepairResult:
        repaired = circuit.copy()
        diagnostics: list[RepairDiagnostic] = []
        applied_fixes: list[str] = []

        if self._mode == "statevector":
            repaired = self._strip_final_measurements(
                repaired, diagnostics, applied_fixes
            )
            if _has_measurements(repaired):
                raise CircuitRepairError(
                    "Statevector mode cannot repair mid-circuit measurements"
                )
        else:
            repaired = self._ensure_measurements(repaired, diagnostics, applied_fixes)

        if self._basis_gates is not None:
            repaired = self._transpile_to_basis(repaired, diagnostics, applied_fixes)

        return RepairResult(
            circuit=repaired,
            diagnostics=tuple(diagnostics),
            applied_fixes=tuple(applied_fixes),
        )

    def _strip_final_measurements(
        self,
        circuit: QuantumCircuit,
        diagnostics: list[RepairDiagnostic],
        applied_fixes: list[str],
    ) -> QuantumCircuit:
        measure_count = circuit.count_ops().get("measure", 0)
        stripped = circuit.remove_final_measurements(inplace=False)
        assert stripped is not None
        stripped_measure_count = stripped.count_ops().get("measure", 0)
        if stripped_measure_count < measure_count:
            _record(
                diagnostics,
                applied_fixes,
                "STRIPPED_FINAL_MEASUREMENTS",
                "Removed final measurements so the circuit can be used as a statevector.",
            )
        return stripped

    def _ensure_measurements(
        self,
        circuit: QuantumCircuit,
        diagnostics: list[RepairDiagnostic],
        applied_fixes: list[str],
    ) -> QuantumCircuit:
        if _has_measurements(circuit):
            return circuit
        measured = circuit.measure_all(inplace=False)
        assert measured is not None
        _record(
            diagnostics,
            applied_fixes,
            "ADDED_MEASUREMENTS",
            "Added measurements for all qubits so the circuit can be sampled.",
        )
        return measured

    def _transpile_to_basis(
        self,
        circuit: QuantumCircuit,
        diagnostics: list[RepairDiagnostic],
        applied_fixes: list[str],
    ) -> QuantumCircuit:
        if self._basis_gates is None:
            return circuit
        pass_manager = generate_preset_pass_manager(
            optimization_level=self._optimization_level,
            basis_gates=list(self._basis_gates),
        )
        transpiled = pass_manager.run(circuit)
        _record(
            diagnostics,
            applied_fixes,
            "TRANSPILED_TO_BASIS",
            "Transpiled the circuit to the requested basis gates.",
        )
        return transpiled


def repair_quantum_circuit(
    circuit: QuantumCircuit,
    *,
    mode: CircuitExecutionMode = "statevector",
    basis_gates: Sequence[str] | None = None,
    optimization_level: int = 1,
) -> RepairResult:
    return QuantumProgramRepairer(
        mode=mode,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
    ).repair(circuit)


def _has_measurements(circuit: QuantumCircuit) -> bool:
    return bool(circuit.count_ops().get("measure", 0))


def _record(
    diagnostics: list[RepairDiagnostic],
    applied_fixes: list[str],
    code: str,
    message: str,
) -> None:
    diagnostics.append(RepairDiagnostic(code=code, message=message))
    applied_fixes.append(code)
