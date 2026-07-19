from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from double_quant.algorithm.circuit import visualize_quantum_circuit  # noqa: E402
from double_quant.algorithm.shapley import QuantumShapleyCalculator  # noqa: E402
from double_quant.application.risk import RiskSavingValueFunction  # noqa: E402


DOC_DIR = Path(__file__).resolve().parents[2] / "docs" / "83-circuit-visualization"
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
    circuit, _ = calculator.build_player_circuit(target_player=0)
    return circuit


def test_quantum_circuit_visualization_exports_quantum_shapley_risk_circuit():
    output_path = IMAGE_DIR / "quantum_shapley_risk_circuit.png"
    result = visualize_quantum_circuit(
        _quantum_shapley_risk_circuit(),
        output_path=output_path,
        title="量子电路可视化：量子 Shapley 风险归因",
        fold=-1,
        scale=0.75,
    )

    print("\n[量子电路可视化] 量子 Shapley 风险归因电路")
    print(f"导出图片：{output_path}")
    print("文本电路：")
    print(result.text_diagram)
    print(f"门操作统计：{dict(result.circuit.count_ops())}")

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert result.circuit.num_qubits == 5
    assert result.circuit.count_ops()["state_preparation"] == 1
    assert result.circuit.count_ops()["ucry"] == 1
    assert result.circuit.count_ops()["cry"] == 4
