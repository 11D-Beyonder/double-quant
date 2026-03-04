from collections import defaultdict
from itertools import combinations
import numpy as np
import pytest

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from scipy import integrate, special

from double_quant.algorithm.shapley import (
    BinaryEnumerationCalculator,
    IntervalLoader,
    PermutationEnumerationCalculator,
    QuantumShapleyCaculator,
    ValueLoader,
    VertexRotator,
)


def generate_value_function(n: int, upper_gain_limit: int = 5):
    """生成每个子集的价值。
    用二进制表示一个子集 S，比如编号为 0 的玩家在子集中，则二进制第 0 位为 1。

    Args:
        n (int): 玩家数。
        upper_gain_limit (int, optional): 单个玩家加入一个子集后，收益增加的上限值，默认为5（开区间）。

    Returns:
        value (dict[int, int]): 子集 S 的收益为 value[S]，满足超加和博弈。
        S 在字典中是用十进制存的，需要用按位与解析。
    """
    contribution = defaultdict(int)
    for size in range(1, n + 1):
        for subset in combinations(range(n), size):
            bin_set = 0
            for idx in subset:
                bin_set |= 1 << idx
            for idx in range(n):
                if (bin_set & (1 << idx)) != 0:
                    contribution[bin_set] = max(
                        contribution[bin_set],
                        contribution[bin_set ^ (1 << idx)]
                        + np.random.randint(upper_gain_limit),
                    )
    return contribution


def _t_func(ell: int, k: int):
    assert 0 <= k <= 2**ell, "k out of range"
    return np.sin(np.pi * k / 2 ** (ell + 1)) ** 2


class TestMath:
    def test_integral_and_weight(self):
        """基于二进制枚举方法的权重，测试其值是否等于一个定积分。"""
        for n in range(2, 133):
            for s in range(n):
                weight = float(1 / n / special.comb(n - 1, s))
                integral = integrate.quad(
                    lambda x: (x**s) * ((1 - x) ** (n - s - 1)), 0, 1
                )[0]
                assert weight == pytest.approx(integral)

    def test_interval_point(self):
        for ell in range(3, 14):
            for k in range(2**ell):
                a = _t_func(ell, k)
                mid = _t_func(ell + 1, 2 * k + 1)
                b = _t_func(ell, k + 1)
                assert a < mid
                assert mid < b


class TestUtil:
    def test_contribution_preparation(self):
        """检查构造的数据是否满足超加和性。
        即 V(S ∪ {i}) ⩾ V(S)
        """
        for _ in range(100):
            for n in range(2, 10):
                contribution = generate_value_function(n)
                for bin_set in range(2**n):
                    for idx in range(n):
                        if bin_set & (1 << idx):
                            assert (
                                contribution[bin_set]
                                >= contribution[bin_set ^ (1 << idx)]
                            )


class TestCalculator:
    def test_methods(self):
        """对拍三个算法。
        1. 基于枚举排列，对每个排列取以 i 为结尾的前缀作为子集。
        2. 基于二进制枚举集合 S。
        3. QCE 2023 量子算法
        """
        for n in range(2, 10):
            data = generate_value_function(n)
            res1 = PermutationEnumerationCalculator(n, data).get_all()
            res2 = BinaryEnumerationCalculator(n, data).get_all()
            res3 = QuantumShapleyCaculator(n, data, internal_multiplier=3).get_all()
            assert np.allclose(res1, res2, 1.0e-5, 1.0e-8)
            assert np.allclose(res2, res3, 1.0e-3, 1.0e-2)


class TestCircuit:
    def test_interval_loader(self):
        for num_qubits in range(2, 11):
            simulator = AerSimulator(method="statevector")
            circuit = IntervalLoader(num_qubits)
            circuit.save_statevector()
            circuit = generate_preset_pass_manager(3, simulator).run(circuit)
            result = simulator.run(circuit).result().get_statevector()
            result = np.real(result.data)

            k_range = np.arange(2**num_qubits + 1)
            vertexes = np.sin(np.pi * k_range / 2 ** (num_qubits + 1)) ** 2
            weights = vertexes[1:] - vertexes[:-1]
            # 每个区间的概率分布要一致
            assert np.allclose(result, np.sqrt(weights))

    def test_vertex_rotator(self):
        for num_control in range(2, 11):
            for k in range(2**num_control):
                init_state = np.zeros(2**num_control)
                init_state[k] = 1
                circuit = QuantumCircuit(num_control + 1)
                circuit.initialize(init_state, range(num_control))
                circuit.append(VertexRotator(num_control), range(num_control + 1))

                t_value = _t_func(num_control + 1, k * 2 + 1)

                # 经过旋转门之后，受控比特 0 态的概率应该为 1 - t_value，1 态的概率应该为 t_value。
                assert np.allclose(
                    Statevector(circuit).probabilities([num_control]),
                    [1 - t_value, t_value],
                )

    def test_value_loader(self):
        def _get_target_probabilities(
            num_control: int,
            values: np.ndarray | list[float] | float,
            init_state: np.ndarray | list[float],
        ):
            circuit = QuantumCircuit(num_control + 1)
            circuit.initialize(init_state, list(range(num_control)))
            circuit.append(ValueLoader(values, num_control), range(num_control + 1))
            # 得到受控比特的态矢量
            return Statevector(circuit).probabilities([num_control])

        for num_control in range(2, 9):
            for h in range(2**num_control):
                init_state = np.zeros(2**num_control)
                init_state[h] = 1

                # 特例：values 每个元素都相等，标准化后全为 1。
                values = (np.random.random() + 1) * 100
                assert np.allclose(
                    _get_target_probabilities(num_control, values, init_state),
                    [0, 1],
                )

                values = (np.random.random(2**num_control) + 1) * 100
                normed_values = values / np.max(values)
                assert np.allclose(
                    _get_target_probabilities(num_control, normed_values, init_state),
                    [1 - normed_values[h], normed_values[h]],
                )
