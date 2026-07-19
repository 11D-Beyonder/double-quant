from __future__ import annotations

from collections.abc import Callable, Mapping

from simulator.metrics import (
    bitstring_objective_accuracy,
    distribution_total_variation,
    fidelity,
    solution_success_probability,
)
from simulator.models import (
    CapacityReport,
    ComplexityReport,
    PrecisionReport,
    SimulationBackend,
)


def analyze_complexity(
    circuit,
    *,
    optimization_level: int = 0,
    basis_gates: list[str] | None = None,
) -> ComplexityReport:
    """Measure circuit gate count and depth after optional transpiler optimization."""

    from qiskit import transpile

    if optimization_level not in (0, 1, 2, 3):
        raise ValueError("optimization_level must be 0, 1, 2, or 3")
    transpiled = transpile(
        circuit,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
    )
    operations = {name: int(count) for name, count in transpiled.count_ops().items()}
    two_qubit_gate_count = sum(
        count for name, count in operations.items() if name in {"cx", "cz", "swap"}
    )
    return ComplexityReport(
        num_qubits=transpiled.num_qubits,
        depth=int(transpiled.depth() or 0),
        gate_count=int(sum(operations.values())),
        operations=operations,
        two_qubit_gate_count=int(two_qubit_gate_count),
        optimization_level=optimization_level,
    )


def compare_complexity(
    circuit, *, basis_gates: list[str] | None = None
) -> dict[int, ComplexityReport]:
    """Compare gate count and circuit depth for optimization levels 0 through 3."""

    return {
        level: analyze_complexity(
            circuit, optimization_level=level, basis_gates=basis_gates
        )
        for level in range(4)
    }


def analyze_precision(
    circuit,
    *,
    noise_config=None,
    ideal_distribution: Mapping[str, float] | None = None,
    ideal_statevector=None,
    optimal_bitstrings: set[str] | list[str] | tuple[str, ...] | None = None,
    objective: Callable[[str], float] | None = None,
    optimum_value: float | None = None,
) -> PrecisionReport:
    """Analyze noiseless/noisy fidelity and problem-level solution accuracy."""

    from simulator.backends import simulate_counts, simulate_statevector

    noiseless = simulate_statevector(circuit)
    noisy = simulate_counts(circuit, noise_config) if noise_config else None
    reference_distribution = ideal_distribution or noiseless.probabilities

    noiseless_fidelity = (
        fidelity(ideal_statevector, noiseless.statevector)
        if ideal_statevector is not None and noiseless.statevector is not None
        else None
    )
    noisy_fidelity = None
    total_variation_distance = None
    measured_distribution = noiseless.probabilities
    if noisy is not None and noisy.probabilities is not None:
        measured_distribution = noisy.probabilities
        if reference_distribution is not None:
            total_variation_distance = distribution_total_variation(
                reference_distribution, noisy.probabilities
            )

    success_probability = None
    if optimal_bitstrings is not None and measured_distribution is not None:
        success_probability = solution_success_probability(
            measured_distribution, optimal_bitstrings
        )

    objective_error = None
    if (
        objective is not None
        and optimum_value is not None
        and measured_distribution is not None
    ):
        objective_error = bitstring_objective_accuracy(
            measured_distribution, objective, optimum_value
        )

    return PrecisionReport(
        noiseless_fidelity=noiseless_fidelity,
        noisy_fidelity=noisy_fidelity,
        total_variation_distance=total_variation_distance,
        success_probability=success_probability,
        objective_error=objective_error,
        metadata={
            "num_qubits": circuit.num_qubits,
            "noisy_backend": noisy.backend if noisy is not None else None,
        },
    )


def verify_capacity(
    num_qubits: int,
    backend: SimulationBackend = SimulationBackend.STATEVECTOR_CPU,
    *,
    minimum_required_qubits: int = 20,
) -> CapacityReport:
    """Validate whether a backend selection satisfies the 20+ qubit requirement."""

    if num_qubits < 1:
        raise ValueError("num_qubits must be positive")
    if minimum_required_qubits < 1:
        raise ValueError("minimum_required_qubits must be positive")
    supported = num_qubits >= minimum_required_qubits and backend in {
        SimulationBackend.STATEVECTOR_CPU,
    }
    if supported:
        reason = "backend supports 20+ qubit quantum-finance simulation path"
    elif num_qubits < minimum_required_qubits:
        reason = f"requested {num_qubits} qubits, requires >= {minimum_required_qubits}"
    else:
        reason = f"backend {backend} is not a 20+ qubit capacity backend"
    return CapacityReport(
        backend=backend,
        requested_qubits=num_qubits,
        supported=supported,
        minimum_required_qubits=minimum_required_qubits,
        reason=reason,
    )


def run_capacity_smoke_circuit(
    num_qubits: int = 20,
    backend: SimulationBackend = SimulationBackend.STATEVECTOR_CPU,
):
    """Build and simulate a shallow 20+ qubit finance-style smoke circuit."""

    from simulator.backends import (
        SimulationConfig,
        simulate_counts,
        simulate_statevector,
    )
    from simulator.finance import build_weighted_sum_circuit

    circuit = build_weighted_sum_circuit(num_qubits)
    if backend == SimulationBackend.STATEVECTOR_CPU:
        return simulate_statevector(circuit)
    return simulate_counts(circuit, SimulationConfig())
