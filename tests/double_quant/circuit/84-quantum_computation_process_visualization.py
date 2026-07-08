from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from double_quant.algorithm.circuit import (  # noqa: E402
    visualize_quantum_computation_process,
)
from double_quant.algorithm.hhl import HHLSolver  # noqa: E402


DOC_DIR = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "84-computation-process-visualization"
)
IMAGE_DIR = DOC_DIR / "images"


def _portfolio_hhl_circuit():
    expected_returns = np.array([0.08, 0.12])
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    target_return = 0.10

    matrix = np.zeros((4, 4), dtype=float)
    matrix[0, 2:] = expected_returns
    matrix[1, 2:] = 1.0
    matrix[2:, 0] = expected_returns
    matrix[2:, 1] = 1.0
    matrix[2:, 2:] = covariance

    vector = np.array([target_return, 1.0, 0.0, 0.0], dtype=float)
    return HHLSolver.build_circuit(matrix, vector, max_qpe_qubits=3)


def test_quantum_computation_process_visualizes_portfolio_hhl_algorithm():
    output_path = IMAGE_DIR / "portfolio_hhl_process_snapshot.png"
    animation_path = IMAGE_DIR / "portfolio_hhl_computation_process.gif"
    result = visualize_quantum_computation_process(
        _portfolio_hhl_circuit(),
        output_path=output_path,
        animation_path=animation_path,
        title="量子计算过程可视化：HHL 组合优化过程",
        tracked_qubits=(0, 2, 5),
        max_basis_states=12,
        fps=1,
    )

    print("\n[量子计算过程可视化] HHL 组合优化过程")
    print(f"静态图：{output_path}")
    print(f"GIF 动图：{animation_path}")
    print(f"操作序列：{result.operation_labels}")
    print(f"最终非零概率：{dict(result.final_probabilities)}")

    assert output_path.exists()
    assert animation_path.exists()
    assert output_path.stat().st_size > 0
    assert animation_path.stat().st_size > 0
    assert result.operation_labels[0].startswith("1: State Preparation")
    assert any("QPE" in label for label in result.operation_labels)
    assert np.isclose(sum(result.final_probabilities.values()), 1.0)
    assert len(result.steps) == 5
