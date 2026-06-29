from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate


def build_shor_period_finding_circuit(
    modulus: int,
    *,
    base: int = 2,
    phase_qubits: int | None = None,
    work_qubits: int | None = None,
) -> QuantumCircuit:
    """Build a Shor period-finding circuit for multiplication by ``base mod N``.

    Args:
        modulus: Odd composite integer ``N`` to factor.
        base: Coprime integer ``a`` whose order modulo ``N`` is estimated.
        phase_qubits: Number of phase-estimation qubits.
        work_qubits: Number of work-register qubits. Defaults to ``ceil(log2(N))``.

    Returns:
        A Qiskit circuit implementing controlled modular multiplication and
        inverse QFT phase readout.
    """
    phase_size, work_size = _resolve_register_sizes(modulus, phase_qubits, work_qubits)
    _validate_period_finding_inputs(modulus, base, phase_size, work_size)

    circuit = QuantumCircuit(
        phase_size + work_size,
        phase_size,
        name=f"shor_N{modulus}_a{base}",
    )
    phase_register = list(range(phase_size))
    work_register = list(range(phase_size, phase_size + work_size))

    circuit.h(phase_register)
    circuit.x(work_register[0])

    qft_control_distance: int | None = None
    qft_swaps = True
    if modulus == 15 and base % modulus == 2:
        _append_optimized_mod15_power_sequence(circuit, phase_register, work_register)
        variant = "optimized_swap_period_15"
        qft_control_distance = None
        qft_swaps = True
    else:
        for phase_index, control in enumerate(phase_register):
            multiplier = pow(base, 2**phase_index, modulus)
            if multiplier == 1:
                continue
            gate = _modular_multiplication_gate(multiplier, modulus, work_size)
            circuit.append(gate.control(1), [control, *work_register])
        variant = "controlled_modular_multiplication_skip_identity"

    _append_inverse_qft(
        circuit,
        phase_register,
        max_control_distance=qft_control_distance,
        do_swaps=qft_swaps,
    )
    circuit.measure(phase_register, phase_register)
    circuit.metadata = {
        "algorithm": "shor_period_finding",
        "variant": variant,
        "modulus": modulus,
        "base": base,
        "phase_qubits": phase_size,
        "work_qubits": work_size,
        "qft_control_distance": "full" if qft_control_distance is None else qft_control_distance,
        "qft_swaps": qft_swaps,
    }
    return circuit


def build_generic_shor_period_finding_circuit(
    modulus: int,
    *,
    base: int = 2,
    phase_qubits: int | None = None,
    work_qubits: int | None = None,
) -> QuantumCircuit:
    """Build an unoptimized generic Shor period-finding circuit."""
    phase_size, work_size = _resolve_register_sizes(modulus, phase_qubits, work_qubits)
    _validate_period_finding_inputs(modulus, base, phase_size, work_size)

    circuit = QuantumCircuit(
        phase_size + work_size,
        phase_size,
        name=f"shor_generic_N{modulus}_a{base}",
    )
    phase_register = list(range(phase_size))
    work_register = list(range(phase_size, phase_size + work_size))

    circuit.h(phase_register)
    circuit.x(work_register[0])

    for phase_index, control in enumerate(phase_register):
        multiplier = pow(base, 2**phase_index, modulus)
        if multiplier == 1:
            # Keep an unoptimized identity-power block as U U^†. This is
            # functionally identity, but it reflects a baseline that does not
            # pre-simplify modular exponentiation powers before circuit synthesis.
            work_gate = _modular_multiplication_gate(base, modulus, work_size)
            circuit.append(work_gate.control(1), [control, *work_register])
            circuit.append(work_gate.inverse().control(1), [control, *work_register])
            continue
        gate = _modular_multiplication_gate(multiplier, modulus, work_size)
        circuit.append(gate.control(1), [control, *work_register])

    _append_inverse_qft(circuit, phase_register)
    circuit.measure(phase_register, phase_register)
    circuit.metadata = {
        "algorithm": "shor_period_finding",
        "variant": "generic_permutation_baseline",
        "modulus": modulus,
        "base": base,
        "phase_qubits": phase_size,
        "work_qubits": work_size,
    }
    return circuit


def build_unoptimized_shor_period_finding_circuit(
    modulus: int,
    *,
    base: int = 2,
    phase_qubits: int | None = None,
    work_qubits: int | None = None,
) -> QuantumCircuit:
    """Build a quantum baseline using generic modular-permutation gates."""
    from double_quant.algorithm.shor.baseline import (
        build_unoptimized_shor_period_finding_circuit as build_baseline,
    )

    return build_baseline(
        modulus,
        base=base,
        phase_qubits=phase_qubits,
        work_qubits=work_qubits,
    )


def _resolve_register_sizes(
    modulus: int,
    phase_qubits: int | None,
    work_qubits: int | None,
) -> tuple[int, int]:
    if modulus <= 2:
        raise ValueError("modulus must be greater than 2")
    work_size = work_qubits or math.ceil(math.log2(modulus))
    phase_size = phase_qubits or 2 * work_size
    return phase_size, work_size


def _validate_period_finding_inputs(
    modulus: int,
    base: int,
    phase_qubits: int,
    work_qubits: int,
) -> None:
    if phase_qubits <= 0:
        raise ValueError("phase_qubits must be positive")
    if work_qubits <= 0:
        raise ValueError("work_qubits must be positive")
    if not (1 < base < modulus):
        raise ValueError("base must satisfy 1 < base < modulus")
    if math.gcd(base, modulus) != 1:
        raise ValueError("base and modulus must be coprime")
    if 2**work_qubits <= modulus:
        raise ValueError("work register is too small for modulus")


def _append_optimized_mod15_power_sequence(
    circuit: QuantumCircuit,
    phase_register: list[int],
    work_register: list[int],
) -> None:
    # For N=15 and a=2, only the first two powers are non-identity:
    # x -> 2x mod 15 and x -> 4x mod 15. Higher powers are identity because r=4.
    if len(work_register) < 4:
        raise ValueError("N=15 optimized modular multiplication needs four work qubits")
    for phase_index, control in enumerate(phase_register):
        power_mod = 2 ** ((2**phase_index) % 4) % 15
        if phase_index >= 2 or power_mod == 1:
            continue
        if power_mod == 2:
            _append_controlled_swap_cycle(circuit, control, work_register, (0, 1, 2, 3))
        elif power_mod == 4:
            circuit.cswap(control, work_register[0], work_register[2])
            circuit.cswap(control, work_register[1], work_register[3])
        else:
            gate = _modular_multiplication_gate(power_mod, 15, len(work_register))
            circuit.append(gate.control(1), [control, *work_register])


def _append_controlled_swap_cycle(
    circuit: QuantumCircuit,
    control: int,
    work_register: list[int],
    cycle: tuple[int, ...],
) -> None:
    first = cycle[0]
    for target in cycle[1:]:
        circuit.cswap(control, work_register[first], work_register[target])


def _modular_multiplication_gate(
    multiplier: int,
    modulus: int,
    work_qubits: int,
) -> UnitaryGate:
    dimension = 2**work_qubits
    matrix = np.zeros((dimension, dimension), dtype=complex)
    for state in range(dimension):
        mapped_state = (multiplier * state) % modulus if state < modulus else state
        matrix[mapped_state, state] = 1.0
    return UnitaryGate(matrix, label=f"*{multiplier} mod {modulus}")


def _append_inverse_qft(
    circuit: QuantumCircuit,
    qubits: Sequence[int],
    *,
    max_control_distance: int | None = None,
    do_swaps: bool = True,
) -> None:
    if do_swaps:
        for left, right in zip(qubits[: len(qubits) // 2], reversed(qubits)):
            circuit.swap(left, right)

    for target_index, target in enumerate(qubits):
        for control_index in range(target_index):
            distance = target_index - control_index
            if max_control_distance is not None and distance > max_control_distance:
                continue
            angle = -math.pi / 2 ** distance
            circuit.cp(angle, qubits[control_index], target)
        circuit.h(target)
