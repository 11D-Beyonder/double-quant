import numpy as np
from dataclasses import dataclass
from itertools import permutations
from typing import Literal, Protocol

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import BlueprintCircuit, StatePreparation, UCRYGate
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.backends import AerSimulator
from qiskit_algorithms import (
    AmplitudeEstimation,
    EstimationProblem,
    IterativeAmplitudeEstimation,
    MaximumLikelihoodAmplitudeEstimation,
)
from scipy import special

from double_quant.common.util import normalize

# Amplitude extraction mode for QuantumCalculator.
# - "statevector": exact extraction via Statevector simulation (1 oracle call)
# - "shots": shot-based sampling via StatevectorSampler (oracle calls = shots)
# - "qae_canonical": canonical QAE using Quantum Phase Estimation
# - "qae_iqae": Iterative QAE — no ancilla, adaptive Grover iterations
# - "qae_mlqae": Maximum-Likelihood QAE — multi-depth circuits + MLE post-processing
ExtractionMode = Literal[
    "statevector", "shots", "qae_canonical", "qae_iqae", "qae_mlqae"
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


class ValueFunction(Protocol):
    """Protocol for characteristic value functions used in Shapley calculations.

    Any object implementing ``__getitem__(bitmask: int) -> float`` satisfies
    this protocol, including plain ``dict[int, int | float]`` instances and
    custom callable value-function classes.
    """

    def __getitem__(self, bitmask: int) -> float: ...


class ControlledBlueprintCircuit(BlueprintCircuit):
    def __init__(self, num_control: int, name: str = "ControlledBlueprintCircuit"):
        super().__init__(name=name)
        self._num_control = 0
        self.num_control = num_control

    @property
    def num_control(self):
        return self._num_control

    @property
    def num_qubits(self):
        return self.num_control + 1

    @num_control.setter
    def num_control(self, num_control: int):
        if num_control != self.num_control:
            self._invalidate()
            self._num_control = num_control
            self.qregs: list[QuantumRegister] = [
                QuantumRegister(num_control, "q"),
                QuantumRegister(1, "target"),
            ]

    def _check_configuration(self, raise_on_failure=True):
        valid = True
        if self.num_control is None or self.num_control <= 0:
            valid = False
            if raise_on_failure:
                raise AttributeError("Number of qubits must be non-negative.")
        if len(self.qregs) != 2:
            valid = False
            if raise_on_failure:
                raise AttributeError("The number of quanum registers must be 2.")
        return valid


class IntervalLoader(BlueprintCircuit):
    """制备描述 $[0, 1]$ 内 $2^{l}$ 个区间的量子态。"""

    def __init__(self, num_qubits: int, name: str = "IntervalLoader"):
        super().__init__(name=name)
        self.num_qubits = num_qubits

    @property
    def num_qubits(self):
        return super().num_qubits

    @num_qubits.setter
    def num_qubits(self, num_qubits: int):
        if num_qubits != self.num_qubits:
            self._invalidate()
            self.qregs = []
            if num_qubits is not None and num_qubits > 0:
                self.qregs = [QuantumRegister(num_qubits, name="q")]

    def _check_configuration(self, raise_on_failure=True):
        valid = True
        if self.num_qubits is None or self.num_qubits <= 0:
            valid = False
            if raise_on_failure:
                raise AttributeError("Number of qubits must be non-negative.")
        if len(self.qregs) != 1:
            valid = False
            if raise_on_failure:
                raise AttributeError("The number of quanum registers must be 1.")
        return valid

    def _build(self):
        if self._is_built:
            return
        super()._build()
        k_range = np.arange(2**self.num_qubits + 1)
        vertexes = np.sin(np.pi * k_range / (2 ** (self.num_qubits + 1))) ** 2
        weights = np.sqrt(vertexes[1:] - vertexes[:-1])
        self.append(StatePreparation(weights), self.qregs)


class VertexRotator(ControlledBlueprintCircuit):
    """制备针对单个比特的大旋转门。"""

    def __init__(self, num_control: int, name: str = "VertexRotator"):
        super().__init__(num_control, name)

    def _build(self):
        if self._is_built:
            return
        super()._build()
        for i in reversed(range(self.num_control)):
            self.cry(
                np.pi / 2 ** (self.num_control - i), self.qregs[0][i], self.qregs[1]
            )
        self.ry(np.pi / 2 ** (self.num_control + 1), self.qregs[1])


class ValueLoader(ControlledBlueprintCircuit):
    """用 UCRYGate 制备每个二进制表示的价值函数。"""

    def __init__(
        self,
        values: np.ndarray | list[float] | float | int,
        num_control: int | None = None,
        name: str = "ValueLoader",
        normalization: bool = True,
    ):
        values_array: np.ndarray
        if isinstance(values, float | int):
            if num_control is None:
                raise ValueError(
                    "num_control must be provided when values is a scalar."
                )
            values_array = np.array([values] * (2**num_control))
        elif isinstance(values, list):
            values_array = np.array(values)
        else:
            values_array = values

        if num_control is not None and len(values_array) != 2**num_control:
            raise ValueError("The length of values must be 2**num_control.")
        num_control = int(np.log2(len(values_array)))
        super().__init__(num_control, name)

        if normalization:
            max_val = np.max(values_array)
            if max_val != 0:
                values_array = values_array / max_val

        self._values = values_array

    def _check_configuration(self, raise_on_failure=True) -> bool:
        return (
            super()._check_configuration(raise_on_failure)
            and len(self.values) == 2**self.num_control
        )

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values: np.ndarray | list[float] | float):
        self._invalidate()
        self.__init__(values)

    def _build(self):
        if self._is_built:
            return
        super()._build()
        self.compose(
            UCRYGate((2 * np.arcsin(np.sqrt(self.values))).tolist()),
            [self.num_control] + list(range(self.num_control)),
            inplace=True,
        )


if __name__ == "__main__":
    # NOTE 低位画在电路图上方。
    print(IntervalLoader(3).draw())
    """
        ┌─────────────────────────────────────────────────────────────────────────────────────┐
    q_0:┤0                                                                                    ├
        │                                                                                     │
    q_1:┤1 State Preparation(0.19509,0.32922,0.40276,0.43743,0.43743,0.40276,0.32922,0.19509) ├
        │                                                                                     │
    q_2:┤2                                                                                    ├
        └─────────────────────────────────────────────────────────────────────────────────────┘
    """
    print(VertexRotator(2).draw())
    """
       q_0: ────────────────■────────────────
                            │                
       q_1: ─────■──────────┼────────────────
            ┌────┴────┐┌────┴────┐┌─────────┐
    target: ┤ Ry(π/2) ├┤ Ry(π/4) ├┤ Ry(π/8) ├
            └─────────┘└─────────┘└─────────┘
    """
    print(ValueLoader([1, 2, 3, 4], 2).draw())
    """
            ┌───────────────────────┐
       q_0: ┤1                      ├
            │                       │
       q_1: ┤2 Ucry(π/3,π/2,2π/3,π) ├
            │                       │
    target: ┤0                      ├
            └───────────────────────┘
    """


class ShapleyCalculator:
    """计算 Shapley 值的基类 只需要实现 _calculate_one 方法，该方法计算给定 player 的 Shapley 值。"""

    def __init__(self, num_players: int, value_dict: ValueFunction | None = None):
        """初始化 Shapley 计算器。

        Args:
            num_players (int): 玩家数。
            value_dict (ValueFunction, optional): 子集收益值字典，为 None 时随机生成。
        """
        self.num_players = num_players

        self.value_dict = value_dict

        self._shapley_cache: list[float | None] = [None] * num_players

    def _calculate_one(self, target_player: int) -> float:
        raise NotImplementedError

    def get_one(self, target_player: int) -> float:
        if self._shapley_cache[target_player] is None:
            self._shapley_cache[target_player] = self._calculate_one(target_player)

        val = self._shapley_cache[target_player]
        if val is None:
            raise RuntimeError(
                f"Failed to calculate Shapley value for player {target_player}"
            )
        return val

    def get_all(self) -> list[float]:
        return [self.get_one(i) for i in range(self.num_players)]


class BinaryEnumerationCalculator(ShapleyCalculator):
    def __init__(self, num_players: int, value_dict: ValueFunction | None = None):
        super().__init__(num_players, value_dict)
        self.__factorial = [1] * (num_players + 1)
        for i in range(1, num_players + 1):
            self.__factorial[i] = self.__factorial[i - 1] * i

    def _calculate_one(self, target_player: int):
        if self.value_dict is None:
            raise ValueError("value_dict is required")

        contribution = 0
        for subset in range(2**self.num_players):
            # NOTE: 枚举不包含 player_index 的子集。
            if (1 << target_player) & subset:
                continue
            subset_size = bin(subset).count("1")
            weight = (
                self.__factorial[self.num_players - subset_size - 1]
                / self.__factorial[self.num_players]
                * self.__factorial[subset_size]
            )
            contribution += weight * (
                self.value_dict[subset | (1 << target_player)] - self.value_dict[subset]
            )
        return contribution


class PermutationEnumerationCalculator(ShapleyCalculator):
    def __init__(self, num_players: int, value_dict: ValueFunction | None = None):
        super().__init__(num_players, value_dict)
        self.__weight = special.factorial(self.num_players, exact=True)

    def _calculate_one(self, target_player):
        if self.value_dict is None:
            raise ValueError("value_dict is required")

        contribution = 0
        for subset in permutations(range(self.num_players), self.num_players - 1):
            bin_pre = 0
            for idx in subset:
                if idx == target_player:
                    break
                bin_pre |= 1 << idx
            contribution += (
                self.value_dict[bin_pre | (1 << target_player)]
                - self.value_dict[bin_pre]
            ) / self.__weight
        return contribution


class QuantumCalculator(ShapleyCalculator):
    def __init__(
        self,
        num_players: int,
        value_dict: ValueFunction | None = None,
        internal_qubits_num: int | None = None,
        internal_multiplier: float = 2,
        extraction_mode: ExtractionMode = "statevector",
        options: QAEOptions | None = None,
    ):
        super().__init__(num_players, value_dict)
        if internal_qubits_num is None:
            internal_qubits_num = int(np.log2(num_players))
        if internal_multiplier:
            internal_qubits_num = int(internal_qubits_num * internal_multiplier)
        self.qreg_qubit_map = {
            "internal": np.arange(internal_qubits_num, dtype=int).tolist(),
            "player": np.arange(
                internal_qubits_num,
                internal_qubits_num + num_players - 1,
                dtype=int,
            ).tolist(),
            "output": [internal_qubits_num + num_players - 1],
        }
        self._init_circuit = self._initilizae_circuit()
        self._simulator = AerSimulator(method="statevector")
        self._pass_manager = generate_preset_pass_manager(3, self._simulator)
        self._extraction_mode = extraction_mode
        self._options = options
        self._oracle_call_counts: list[int | None] = [None] * num_players

    def _initilizae_circuit(self):
        circuit = QuantumCircuit(
            len(self.qreg_qubit_map["internal"])
            + len(self.qreg_qubit_map["player"])
            + len(self.qreg_qubit_map["output"]),
        )
        circuit.compose(
            IntervalLoader(len(self.qreg_qubit_map["internal"])),
            self.qreg_qubit_map["internal"],
            inplace=True,
        )
        for q in self.qreg_qubit_map["player"]:
            circuit.compose(
                VertexRotator(len(self.qreg_qubit_map["internal"])),
                self.qreg_qubit_map["internal"] + [q],
                inplace=True,
            )
        return circuit

    def _extend_circuit(self, target_player: int):
        if self.value_dict is None:
            raise ValueError("value_dict is required")

        # h1...h(i-1)<h(hi)>h(i+1)...h(n-1)
        # 改造 value_dict，把 player_index 位为 0 和 1 的位置编码到电路上
        # W(h)=V(S \cup {i})-V(S)
        # Uw(h)=(1-W(h))|0\rangle
        contributions = [0] * (2 ** (self.num_players - 1))
        for bit in range(2**self.num_players):
            if bit & (1 << target_player):
                continue
            value_out = self.value_dict[bit]
            value_in = self.value_dict[bit ^ (1 << target_player)]
            assert value_in >= value_out, "出现负的贡献"
            bin_str = format(bit, "b").zfill(self.num_players)

            bin_str_without_target = (
                bin_str[:-1]
                if target_player == 0
                else bin_str[: -1 - target_player] + bin_str[-target_player:]
            )
            contributions[int(bin_str_without_target, 2)] = value_in - value_out

        contributions_arr, max_contribution = normalize(np.array(contributions))

        circuit = self._init_circuit.compose(
            ValueLoader(contributions_arr, normalization=False),
            self.qreg_qubit_map["player"] + self.qreg_qubit_map["output"],
        )
        return circuit, max_contribution

    def get_oracle_count(self, player_index: int) -> int | None:
        """Return the number of oracle calls used to estimate the amplitude for
        the given player. Returns None if the player's Shapley value has not
        been computed yet."""
        return self._oracle_call_counts[player_index]

    def _run_statevector(
        self, circuit: QuantumCircuit, max_contribution: float
    ) -> tuple[float, int]:
        """Extract amplitude via exact Statevector simulation. Oracle count = 1."""
        output_qubit = self.qreg_qubit_map["output"][0]
        prob = Statevector(self._pass_manager.run(circuit)).probabilities(
            [output_qubit]
        )[1]
        return float(prob * max_contribution), 1

    def _run_shots(
        self, circuit: QuantumCircuit, max_contribution: float
    ) -> tuple[float, int]:
        """Estimate amplitude by shot-based sampling. Oracle count = shots."""
        if self._options is None:
            raise ValueError("QAEOptions required for extraction_mode='shots'")
        shots = self._options.shots
        output_qubit = self.qreg_qubit_map["output"][0]

        meas_circuit = circuit.copy()
        cr = ClassicalRegister(1, "out")
        meas_circuit.add_register(cr)
        meas_circuit.measure(output_qubit, cr[0])

        sampler = StatevectorSampler()
        job = sampler.run([meas_circuit], shots=shots)
        counts = job.result()[0].data.out.get_counts()
        good = counts.get("1", 0)
        return float(good / shots * max_contribution), shots

    def _run_qae(
        self, circuit: QuantumCircuit, max_contribution: float
    ) -> tuple[float, int]:
        """Estimate amplitude via Quantum Amplitude Estimation.

        Dispatches to one of three QAE variants based on self._extraction_mode:
        - "qae_canonical": standard QPE-based QAE
        - "qae_iqae": iterative QAE (no ancilla, adaptive Grover iterations)
        - "qae_mlqae": maximum-likelihood QAE (multi-depth + MLE)
        """
        if self._options is None:
            raise ValueError(
                f"QAEOptions required for extraction_mode='{self._extraction_mode}'"
            )
        opts = self._options
        output_qubit = self.qreg_qubit_map["output"][0]

        problem = EstimationProblem(
            state_preparation=circuit,
            objective_qubits=[output_qubit],
        )
        sampler = StatevectorSampler()

        if self._extraction_mode == "qae_canonical":
            algo: AmplitudeEstimation | IterativeAmplitudeEstimation | MaximumLikelihoodAmplitudeEstimation = AmplitudeEstimation(
                num_eval_qubits=opts.num_eval_qubits,
                sampler=sampler,
            )
        elif self._extraction_mode == "qae_iqae":
            algo = IterativeAmplitudeEstimation(
                epsilon_target=opts.epsilon,
                alpha=opts.alpha,
                sampler=sampler,
            )
        else:  # qae_mlqae
            algo = MaximumLikelihoodAmplitudeEstimation(
                evaluation_schedule=opts.num_eval_qubits,
                sampler=sampler,
            )

        result = algo.estimate(problem)
        return float(result.estimation * max_contribution), result.num_oracle_queries

    def _calculate_one(self, player_index: int) -> float:
        circuit, max_contribution = self._extend_circuit(player_index)
        if circuit is None:
            raise RuntimeError("Failed to extend circuit")

        if self._extraction_mode == "statevector":
            value, count = self._run_statevector(circuit, max_contribution)
        elif self._extraction_mode == "shots":
            value, count = self._run_shots(circuit, max_contribution)
        else:
            value, count = self._run_qae(circuit, max_contribution)

        self._oracle_call_counts[player_index] = count
        return value


"""
QuantumCalculator

n+1

A, B, C, D

0000 a[0]
A 0001 a[1]
B 0010 a[2]
AB 0011 a[3]

50% A 50%B AB

[A,B,C,D,...]

输入：2^(n+1) 维的向量 [000,01,010,...,2^(n+1)-1]
输出：第i个标的的夏普利值 get_one(i)
"""
