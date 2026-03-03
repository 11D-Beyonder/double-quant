from dataclasses import dataclass
from typing import Literal, Protocol

# Amplitude extraction mode for QuantumCalculator.
# - "statevector": exact extraction via Statevector simulation (1 oracle call)
# - "shots": shot-based sampling via StatevectorSampler (oracle calls = shots)
# - "qae_canonical": canonical QAE using Quantum Phase Estimation
# - "qae_iqae": Iterative QAE — no ancilla, adaptive Grover iterations
# - "qae_mlqae": Maximum-Likelihood QAE — multi-depth circuits + MLE post-processing
ExtractionMode = Literal[
    "statevector", "shots", "qae_canonical", "qae_iqae", "qae_mlqae", "qae_fae"
]


@dataclass
class QAEOptions:
    """Parameters controlling the amplitude extraction strategy of QuantumCalculator.

    Fields are mode-specific; unused fields for a given mode are ignored.
    """

    # "shots" mode: number of measurement shots
    shots: int = 1024
    # All QAE modes: target precision (half confidence-interval width)
    epsilon: float = 0.01
    # IQAE / MLQAE: confidence level; probability of exceeding epsilon is <= alpha
    alpha: float = 0.05
    # "qae_canonical": number of QPE evaluation qubits; oracle calls ~ 2^num_eval_qubits
    # "qae_mlqae": evaluation schedule depth; oracle calls ~ 2^num_eval_qubits
    num_eval_qubits: int = 3
    # "qae_fae": confidence parameter and max iterations
    delta: float = 0.05
    maxiter: int = 5


class ValueFunction(Protocol):
    """Protocol for characteristic value functions used in Shapley calculations.

    Any object implementing ``__getitem__(bitmask: int) -> float`` satisfies
    this protocol, including plain ``dict[int, int | float]`` instances and
    custom callable value-function classes.
    """

    def __getitem__(self, bitmask: int) -> float: ...
