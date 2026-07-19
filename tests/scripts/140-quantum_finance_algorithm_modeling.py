from pathlib import Path
import sys

from qiskit import QuantumCircuit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simulator.finance import build_portfolio_qaoa_ansatz, build_weighted_sum_circuit


def main() -> None:
    weighted = build_weighted_sum_circuit(4, weights=[0.2, 0.4, 0.6, 0.8])
    portfolio = build_portfolio_qaoa_ansatz(
        expected_returns=[0.05, 0.08, 0.03],
        covariance=[
            [0.10, 0.02, 0.01],
            [0.02, 0.12, 0.03],
            [0.01, 0.03, 0.09],
        ],
    )
    assert isinstance(weighted, QuantumCircuit)
    assert isinstance(portfolio, QuantumCircuit)
    assert weighted.num_qubits == 4
    assert portfolio.num_qubits == 3
    print("140 仿真工具支持量子金融算法建模: PASS")
    print(f"资产权重求和电路使用的量子比特数：{weighted.num_qubits}")
    print(f"资产权重求和电路深度：{weighted.depth()}")
    print(f"投资组合 QAOA 电路使用的量子比特数：{portfolio.num_qubits}")
    print(f"投资组合 QAOA 电路深度：{portfolio.depth()}")


if __name__ == "__main__":
    main()
