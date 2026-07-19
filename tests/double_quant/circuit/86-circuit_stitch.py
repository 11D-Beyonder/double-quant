import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from double_quant.algorithm.circuit import CircuitStitchingError, stitch_circuits


def _format_qubit_map(qubit_map: dict[int, int]) -> str:
    if not qubit_map:
        return "无量子比特映射"
    return "，".join(
        f"右侧 q[{source}] -> 输出 q[{target}]"
        for source, target in qubit_map.items()
    )


def _print_stitch_case(name: str, left: QuantumCircuit, right: QuantumCircuit, result):
    print(f"\n[智能拼接用例] {name}")
    print("左侧原始电路：")
    print(left)
    print(
        "左侧摘要："
        f"量子比特数={left.num_qubits}，经典比特数={left.num_clbits}，"
        f"门操作={dict(left.count_ops())}"
    )
    print("右侧待拼接电路：")
    print(right)
    print(
        "右侧摘要："
        f"量子比特数={right.num_qubits}，经典比特数={right.num_clbits}，"
        f"门操作={dict(right.count_ops())}"
    )
    print("拼接后电路：")
    print(result.circuit)
    print(
        "拼接后摘要："
        f"量子比特数={result.circuit.num_qubits}，"
        f"经典比特数={result.circuit.num_clbits}，"
        f"门操作={dict(result.circuit.count_ops())}，"
        f"量子比特映射={result.qubit_map}，经典比特映射={result.clbit_map}"
    )
    print(f"量子比特映射明细：{_format_qubit_map(result.qubit_map)}")


def test_stitch_same_width_circuits_matches_manual_compose():
    left = QuantumCircuit(2)
    left.h(0)
    right = QuantumCircuit(2)
    right.cx(0, 1)

    result = stitch_circuits(left, right)
    expected = left.compose(right)
    assert expected is not None

    _print_stitch_case("同线宽自动映射", left, right, result)

    assert result.qubit_map == {0: 0, 1: 1}
    assert Statevector.from_instruction(result.circuit).equiv(
        Statevector.from_instruction(expected)
    )


def test_stitch_with_explicit_qubit_map_matches_manual_circuit():
    left = QuantumCircuit(2)
    left.x(0)
    right = QuantumCircuit(2)
    right.cx(0, 1)

    result = stitch_circuits(left, right, qubit_map={0: 1, 1: 0})
    expected = left.copy()
    expected.compose(right, qubits=[1, 0], inplace=True)

    _print_stitch_case("显式量子比特映射", left, right, result)
    print(
        "显式映射验证：右侧 CX 的控制位 q[0] 被接到输出 q[1]，"
        "目标位 q[1] 被接到输出 q[0]；因此拼接后控制点在 q_1，"
        "目标 X 在 q_0。"
    )

    assert result.qubit_map == {0: 1, 1: 0}
    assert Statevector.from_instruction(result.circuit).equiv(
        Statevector.from_instruction(expected)
    )


def test_stitch_extends_left_circuit_when_allowed():
    left = QuantumCircuit(1)
    left.h(0)
    right = QuantumCircuit(2)
    right.cx(0, 1)

    result = stitch_circuits(left, right, allow_extend=True)
    expected = QuantumCircuit(2)
    expected.h(0)
    expected.cx(0, 1)

    _print_stitch_case("自动扩展输出量子比特", left, right, result)

    assert "EXTENDED_QUBITS" in {diagnostic.code for diagnostic in result.diagnostics}
    assert result.circuit.num_qubits == 2
    assert Statevector.from_instruction(result.circuit).equiv(
        Statevector.from_instruction(expected)
    )


def test_stitch_rejects_incompatible_width_without_extend():
    left = QuantumCircuit(1)
    left.h(0)
    right = QuantumCircuit(2)
    right.cx(0, 1)

    with pytest.raises(CircuitStitchingError) as exc_info:
        stitch_circuits(left, right)

    print("\n[智能拼接用例] 拒绝未授权的线宽扩展")
    print("左侧原始电路：")
    print(left)
    print(
        "左侧摘要："
        f"量子比特数={left.num_qubits}，经典比特数={left.num_clbits}，"
        f"门操作={dict(left.count_ops())}"
    )
    print("右侧待拼接电路：")
    print(right)
    print(
        "右侧摘要："
        f"量子比特数={right.num_qubits}，经典比特数={right.num_clbits}，"
        f"门操作={dict(right.count_ops())}"
    )
    print("拼接参数：allow_extend=False")
    print(f"拒绝原因：{exc_info.value}")
    print("拼接结果：由于 allow_extend=False，已抛出 CircuitStitchingError")
