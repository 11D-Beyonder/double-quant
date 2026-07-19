from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile

from simulator.models import SimulationBackend, SimulationResult


@dataclass(frozen=True, slots=True)
class NoiseConfig:
    """Simple depolarizing noise profile for gate-level noisy simulation."""

    single_qubit_error: float = 0.001
    two_qubit_error: float = 0.01
    readout_error: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("single_qubit_error", self.single_qubit_error),
            ("two_qubit_error", self.two_qubit_error),
            ("readout_error", self.readout_error),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Runtime options for simulator backends."""

    shots: int = 4096
    seed_simulator: int | None = 7
    optimization_level: int = 1
    noise: NoiseConfig | None = None

    def __post_init__(self) -> None:
        if self.shots < 1:
            raise ValueError("shots must be positive")
        if self.optimization_level not in (0, 1, 2, 3):
            raise ValueError("optimization_level must be 0, 1, 2, or 3")


def simulate_statevector(
    circuit: QuantumCircuit, config: SimulationConfig | None = None
) -> SimulationResult:
    """Run noiseless CPU statevector simulation."""

    config = config or SimulationConfig()
    simulator = _aer_simulator(method="statevector", seed_simulator=config.seed_simulator)
    run_circuit = circuit.remove_final_measurements(inplace=False)
    run_circuit.save_statevector()
    result = simulator.run(run_circuit).result()
    statevector = np.asarray(result.get_statevector(run_circuit), dtype=complex)
    probabilities = _statevector_probabilities(statevector, circuit.num_qubits)
    return SimulationResult(
        backend=SimulationBackend.STATEVECTOR_CPU,
        num_qubits=circuit.num_qubits,
        probabilities=probabilities,
        statevector=statevector,
        metadata={"method": "statevector"},
    )


def simulate_statevector_metadata(
    circuit: QuantumCircuit, config: SimulationConfig | None = None
) -> SimulationResult:
    """Run noiseless statevector simulation without materializing probabilities."""

    config = config or SimulationConfig()
    simulator = _aer_simulator(method="statevector", seed_simulator=config.seed_simulator)
    run_circuit = circuit.remove_final_measurements(inplace=False)
    run_circuit.save_statevector()
    result = simulator.run(run_circuit).result()
    statevector = np.asarray(result.get_statevector(run_circuit), dtype=complex)
    nonzero_amplitudes = int(np.count_nonzero(np.abs(statevector) > 1e-12))
    return SimulationResult(
        backend=SimulationBackend.STATEVECTOR_CPU,
        num_qubits=circuit.num_qubits,
        statevector=statevector,
        metadata={
            "method": "statevector",
            "nonzero_amplitudes": nonzero_amplitudes,
        },
    )


def simulate_tensor_network(
    circuit: QuantumCircuit, config: SimulationConfig | None = None
) -> SimulationResult:
    """Run tensor-network simulation using Aer matrix-product-state method."""

    config = config or SimulationConfig()
    simulator = _aer_simulator(
        method="matrix_product_state", seed_simulator=config.seed_simulator
    )
    run_circuit = circuit.remove_final_measurements(inplace=False)
    run_circuit.save_statevector()
    result = simulator.run(run_circuit).result()
    statevector = np.asarray(result.get_statevector(run_circuit), dtype=complex)
    probabilities = _statevector_probabilities(statevector, circuit.num_qubits)
    return SimulationResult(
        backend=SimulationBackend.TENSOR_NETWORK,
        num_qubits=circuit.num_qubits,
        probabilities=probabilities,
        statevector=statevector,
        metadata={"method": "matrix_product_state"},
    )


def simulate_counts(
    circuit: QuantumCircuit, config: SimulationConfig | None = None
) -> SimulationResult:
    """Run shot-based simulation, optionally with depolarizing noise."""

    config = config or SimulationConfig()
    method = "matrix_product_state" if config.noise else "automatic"
    simulator_kwargs: dict[str, Any] = {
        "method": method,
        "seed_simulator": config.seed_simulator,
    }
    if config.noise is not None:
        simulator_kwargs["noise_model"] = build_depolarizing_noise_model(config.noise)
    simulator = _aer_simulator(**simulator_kwargs)
    run_circuit = circuit.copy()
    if run_circuit.num_clbits == 0:
        run_circuit.measure_all()
    transpiled = transpile(run_circuit, simulator, optimization_level=0)
    result = simulator.run(transpiled, shots=config.shots).result()
    counts = dict(result.get_counts(transpiled))
    backend = (
        SimulationBackend.NOISY_TENSOR_NETWORK
        if config.noise is not None
        else SimulationBackend.SHOT_BASED
    )
    return SimulationResult(
        backend=backend,
        num_qubits=circuit.num_qubits,
        counts=counts,
        probabilities={key: value / config.shots for key, value in counts.items()},
        shots=config.shots,
        metadata={"method": method, "noise": config.noise is not None},
    )


def build_depolarizing_noise_model(noise: NoiseConfig):
    """Build an Aer noise model with 1q/2q depolarizing errors and readout error."""

    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    noise_model = NoiseModel()
    if noise.single_qubit_error > 0:
        one_qubit_error = depolarizing_error(noise.single_qubit_error, 1)
        noise_model.add_all_qubit_quantum_error(
            one_qubit_error, ["id", "sx", "x", "y", "z", "h", "rx", "ry", "rz"]
        )
    if noise.two_qubit_error > 0:
        two_qubit_error = depolarizing_error(noise.two_qubit_error, 2)
        noise_model.add_all_qubit_quantum_error(two_qubit_error, ["cx", "cz", "swap"])
    if noise.readout_error > 0:
        readout = ReadoutError(
            [[1.0 - noise.readout_error, noise.readout_error],
             [noise.readout_error, 1.0 - noise.readout_error]]
        )
        noise_model.add_all_qubit_readout_error(readout)
    return noise_model


def _aer_simulator(**kwargs: Any):
    from qiskit_aer import AerSimulator

    return AerSimulator(**kwargs)


def _statevector_probabilities(
    statevector: np.ndarray, num_qubits: int, cutoff: float = 1.0e-15
) -> dict[str, float]:
    probabilities: dict[str, float] = {}
    for index, amplitude in enumerate(statevector):
        probability = float(abs(amplitude) ** 2)
        if probability > cutoff:
            probabilities[format(index, f"0{num_qubits}b")] = probability
    return probabilities
