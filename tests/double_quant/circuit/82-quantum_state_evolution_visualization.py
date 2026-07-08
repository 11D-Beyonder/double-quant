from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from double_quant.algorithm.circuit import visualize_state_evolution  # noqa: E402
from double_quant.algorithm.shapley import QuantumShapleyCalculator  # noqa: E402
from double_quant.application.risk import RiskSavingValueFunction  # noqa: E402


DOC_DIR = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "82-state-evolution-visualization"
)
IMAGE_DIR = DOC_DIR / "images"


def _risk_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AAPL": [-0.021, 0.012, -0.014, 0.018, -0.026, 0.009, -0.017, 0.011],
            "MSFT": [-0.012, 0.010, -0.009, 0.015, -0.018, 0.007, -0.011, 0.008],
            "TLT": [0.006, -0.003, 0.004, -0.002, 0.005, -0.001, 0.003, -0.002],
        }
    )


def _quantum_shapley_risk_circuit():
    value_function = RiskSavingValueFunction(_risk_returns(), alpha=0.75)
    calculator = QuantumShapleyCalculator(
        3,
        value_function,
        internal_qubits_num=2,
        internal_multiplier=1,
    )
    circuit, max_contribution = calculator.build_player_circuit(target_player=0)
    return circuit, max_contribution


def test_state_evolution_visualizes_quantum_shapley_risk_oracle_as_gif():
    circuit, max_contribution = _quantum_shapley_risk_circuit()
    output_path = IMAGE_DIR / "quantum_shapley_state_snapshot.png"
    animation_path = IMAGE_DIR / "quantum_shapley_state_evolution.gif"

    result = visualize_state_evolution(
        circuit,
        output_path=output_path,
        animation_path=animation_path,
        title="量子态演化可视化：量子 Shapley 风险归因",
        tracked_qubits=(0, 2, 4),
        max_basis_states=12,
        fps=1,
    )

    tracked_qubits = (0, 2, 4)
    final_probability_sum = sum(result.steps[-1].probabilities.values())
    output_probability = sum(
        probability
        for bitstring, probability in result.steps[-1].probabilities.items()
        if bitstring[0] == "1"
    )

    print("\n[量子态演化可视化] 量子 Shapley 风险归因态演化")
    print(f"风险贡献归一化因子：{max_contribution:.8f}")
    print(f"静态图：{output_path}")
    print(f"GIF 动图：{animation_path}")
    print(f"演化步数：{len(result.steps)}")
    print(f"输出量子比特振幅概率 P(输出=1)：{output_probability:.8f}")
    print(f"最终概率归一化和：{final_probability_sum:.8f}")
    print(f"跟踪布洛赫球量子比特：{tracked_qubits}")

    assert output_path.exists()
    assert animation_path.exists()
    assert output_path.stat().st_size > 0
    assert animation_path.stat().st_size > 0
    assert len(result.steps) == len(circuit.data) + 1
    assert np.isclose(final_probability_sum, 1.0)
    assert output_probability > 0.0
