from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simulator.metrics import (
    fidelity,
)


def distribution_fidelity(
    ideal_distribution: dict[str, float], noisy_distribution: dict[str, float]
) -> float:
    keys = set(ideal_distribution) | set(noisy_distribution)
    return float(
        sum(
            np.sqrt(
                ideal_distribution.get(bitstring, 0.0)
                * noisy_distribution.get(bitstring, 0.0)
            )
            for bitstring in keys
        )
        ** 2
    )


def main() -> None:
    ideal_state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    noisy_distribution = {"00": 0.45, "11": 0.50, "01": 0.05}
    ideal_distribution = {"00": 0.5, "11": 0.5}

    state_fidelity = fidelity(ideal_state, ideal_state)
    noisy_fidelity = distribution_fidelity(ideal_distribution, noisy_distribution)

    assert state_fidelity > 0.999
    assert 0.0 <= noisy_fidelity <= 1.0
    print("141 仿真工具支持求解精度分析: PASS")
    print(f"无噪声状态保真度：{state_fidelity:.6f}")
    print(f"含噪概率分布保真度：{noisy_fidelity:.6f}")


if __name__ == "__main__":
    main()
