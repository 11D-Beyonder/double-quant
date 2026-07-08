from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister


@dataclass(frozen=True, slots=True)
class StitchDiagnostic:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class StitchResult:
    circuit: QuantumCircuit
    qubit_map: dict[int, int]
    clbit_map: dict[int, int]
    diagnostics: tuple[StitchDiagnostic, ...]


class CircuitStitchingError(ValueError):
    """Raised when two circuits cannot be stitched with the requested mapping."""


def stitch_circuits(
    left: QuantumCircuit,
    right: QuantumCircuit,
    *,
    qubit_map: Mapping[int, int] | None = None,
    clbit_map: Mapping[int, int] | None = None,
    allow_extend: bool = False,
) -> StitchResult:
    diagnostics: list[StitchDiagnostic] = []
    normalized_qubit_map = _normalize_wire_map(
        source_width=right.num_qubits,
        target_width=left.num_qubits,
        explicit_map=qubit_map,
        allow_extend=allow_extend,
        kind="qubit",
        diagnostics=diagnostics,
    )
    normalized_clbit_map = _normalize_wire_map(
        source_width=right.num_clbits,
        target_width=left.num_clbits,
        explicit_map=clbit_map,
        allow_extend=allow_extend,
        kind="clbit",
        diagnostics=diagnostics,
    )

    required_qubits = _required_width(left.num_qubits, normalized_qubit_map)
    required_clbits = _required_width(left.num_clbits, normalized_clbit_map)

    stitched = left.copy()
    if required_qubits > stitched.num_qubits:
        extra = required_qubits - stitched.num_qubits
        stitched.add_register(QuantumRegister(extra, "stitch_q"))
        diagnostics.append(
            StitchDiagnostic(
                code="EXTENDED_QUBITS",
                message=f"Extended the left circuit by {extra} qubit(s).",
            )
        )
    if required_clbits > stitched.num_clbits:
        extra = required_clbits - stitched.num_clbits
        stitched.add_register(ClassicalRegister(extra, "stitch_c"))
        diagnostics.append(
            StitchDiagnostic(
                code="EXTENDED_CLBITS",
                message=f"Extended the left circuit by {extra} clbit(s).",
            )
        )

    stitched.compose(
        right,
        qubits=[normalized_qubit_map[index] for index in range(right.num_qubits)],
        clbits=[normalized_clbit_map[index] for index in range(right.num_clbits)],
        inplace=True,
    )
    return StitchResult(
        circuit=stitched,
        qubit_map=normalized_qubit_map,
        clbit_map=normalized_clbit_map,
        diagnostics=tuple(diagnostics),
    )


def _normalize_wire_map(
    *,
    source_width: int,
    target_width: int,
    explicit_map: Mapping[int, int] | None,
    allow_extend: bool,
    kind: str,
    diagnostics: list[StitchDiagnostic],
) -> dict[int, int]:
    if source_width == 0:
        return {}

    if explicit_map is None:
        if source_width > target_width and not allow_extend:
            raise CircuitStitchingError(
                f"Right circuit has {source_width} {kind}(s), but left circuit "
                f"has {target_width}; pass allow_extend=True to grow the output."
            )
        diagnostics.append(
            StitchDiagnostic(
                code=f"AUTO_{kind.upper()}_MAP",
                message=f"Mapped right {kind}s to left {kind}s by index.",
            )
        )
        return {index: index for index in range(source_width)}

    normalized = dict(explicit_map)
    expected_keys = set(range(source_width))
    actual_keys = set(normalized)
    if actual_keys != expected_keys:
        raise CircuitStitchingError(
            f"Explicit {kind}_map must contain keys {sorted(expected_keys)}; "
            f"got {sorted(actual_keys)}."
        )
    values = list(normalized.values())
    if len(set(values)) != len(values):
        raise CircuitStitchingError(f"Explicit {kind}_map contains duplicate targets.")
    if any(value < 0 for value in values):
        raise CircuitStitchingError(f"Explicit {kind}_map targets must be non-negative.")
    if values and max(values) >= target_width and not allow_extend:
        raise CircuitStitchingError(
            f"Explicit {kind}_map targets require width {max(values) + 1}, "
            f"but left circuit has {target_width}; pass allow_extend=True."
        )

    diagnostics.append(
        StitchDiagnostic(
            code=f"EXPLICIT_{kind.upper()}_MAP",
            message=f"Used the provided {kind} map.",
        )
    )
    return normalized


def _required_width(current_width: int, wire_map: Mapping[int, int]) -> int:
    if not wire_map:
        return current_width
    return max(current_width, max(wire_map.values()) + 1)
