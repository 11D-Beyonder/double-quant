from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simulator.analysis import run_capacity_smoke_circuit, verify_capacity
from simulator.backends import simulate_statevector_metadata
from simulator.finance import build_weighted_sum_circuit
from simulator.models import SimulationBackend


def main() -> None:
    validation_qubits = 20
    thirty_qubit_run_qubits = 30
    report = verify_capacity(
        validation_qubits,
        SimulationBackend.STATEVECTOR_CPU,
        minimum_required_qubits=20,
    )

    start_20 = perf_counter()
    result_20 = run_capacity_smoke_circuit(
        validation_qubits, SimulationBackend.STATEVECTOR_CPU
    )
    elapsed_20 = perf_counter() - start_20

    start_30 = perf_counter()
    result_30 = simulate_statevector_metadata(
        build_weighted_sum_circuit(thirty_qubit_run_qubits)
    )
    elapsed_30 = perf_counter() - start_30

    assert report.supported
    assert result_20.num_qubits == validation_qubits
    assert result_20.statevector is not None
    assert result_30.num_qubits == thirty_qubit_run_qubits
    assert result_30.statevector is not None
    print("145 仿真工具支撑20比特及以上的量子金融算法仿真: PASS")
    print(f"容量检查是否通过：{report.supported}")
    print(f"容量检查使用的后端：{report.backend}")
    print(f"容量检查请求的量子比特数：{report.requested_qubits}")
    print(f"20比特验收运行使用的后端：{result_20.backend}")
    print(f"20比特验收运行的量子比特数：{result_20.num_qubits}")
    print(f"20比特 statevector 运行耗时（秒）：{elapsed_20:.3f}")
    print(
        "20比特 statevector 复振幅数量："
        f"{len(result_20.statevector)}"
    )
    print(
        "20比特非零概率项数量："
        f"{len(result_20.probabilities or {})}"
    )
    print(f"30比特运行使用的后端：{result_30.backend}")
    print(f"30比特运行的量子比特数：{result_30.num_qubits}")
    print(f"30比特 statevector 运行耗时（秒）：{elapsed_30:.3f}")
    print(
        "30比特 statevector 复振幅数量："
        f"{len(result_30.statevector)}"
    )
    print(
        "30比特非零振幅数量："
        f"{result_30.metadata['nonzero_amplitudes']}"
    )


if __name__ == "__main__":
    main()
