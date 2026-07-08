from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simulator.backends import NoiseConfig, SimulationConfig, simulate_counts
from simulator.finance import build_weighted_sum_circuit
from simulator.models import SimulationBackend


NOISE_CASES = [
    ("低噪声", NoiseConfig(single_qubit_error=0.001, two_qubit_error=0.005)),
    ("中等噪声", NoiseConfig(single_qubit_error=0.003, two_qubit_error=0.010)),
    ("高噪声", NoiseConfig(single_qubit_error=0.010, two_qubit_error=0.030)),
    ("单比特门主导噪声", NoiseConfig(single_qubit_error=0.020, two_qubit_error=0.005)),
    ("双比特门主导噪声", NoiseConfig(single_qubit_error=0.001, two_qubit_error=0.050)),
]


def main() -> None:
    circuit = build_weighted_sum_circuit(4)
    results = []

    for index, (name, noise) in enumerate(NOISE_CASES, start=1):
        result = simulate_counts(
            circuit,
            SimulationConfig(
                shots=256,
                seed_simulator=10 + index,
                noise=noise,
            ),
        )

        assert result.backend == SimulationBackend.NOISY_TENSOR_NETWORK
        assert result.counts is not None
        assert sum(result.counts.values()) == 256
        results.append((index, name, noise, result))

    print("144 仿真工具支持含噪量子金融计算模拟: PASS")
    print(f"噪声配置总数：{len(results)}")
    for index, name, noise, result in results:
        print(f"第{index}组噪声名称：{name}")
        print(f"第{index}组单比特门退极化错误率：{noise.single_qubit_error:.3f}")
        print(f"第{index}组双比特门退极化错误率：{noise.two_qubit_error:.3f}")
        print(f"第{index}组含噪仿真使用的后端：{result.backend}")
        print(f"第{index}组采样次数：{result.shots}")
        print(f"第{index}组非零测量态数量：{len(result.counts or {})}")


if __name__ == "__main__":
    main()
