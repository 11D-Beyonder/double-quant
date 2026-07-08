import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Statevector

from double_quant.algorithm.circuit import repair_quantum_circuit


_FIX_LABELS = {
    "TRANSPILED_TO_BASIS": "门集转译",
    "STRIPPED_FINAL_MEASUREMENTS": "移除末尾测量",
    "ADDED_MEASUREMENTS": "补充抽样测量",
}


def _format_fixes(fixes) -> list[str]:
    return [f"{_FIX_LABELS.get(fix, fix)}（{fix}）" for fix in fixes]


def _print_repair_case(name: str, original: QuantumCircuit, repaired: QuantumCircuit, fixes):
    print(f"\n[自动修正用例] {name}")
    print("修正前电路：")
    print(original)
    print(
        "修正前摘要："
        f"量子比特数={original.num_qubits}，经典比特数={original.num_clbits}，"
        f"门操作={dict(original.count_ops())}"
    )
    print("修正后电路：")
    print(repaired)
    print(
        "修正后摘要："
        f"量子比特数={repaired.num_qubits}，经典比特数={repaired.num_clbits}，"
        f"门操作={dict(repaired.count_ops())}，修正项={_format_fixes(fixes)}"
    )


def test_repair_transpiles_unsupported_basis_gates():
    circuit = QuantumCircuit(1)
    circuit.h(0)

    result = repair_quantum_circuit(
        circuit,
        mode="statevector",
        basis_gates=["u3"],
        optimization_level=0,
    )

    _print_repair_case("门集不兼容修正", circuit, result.circuit, result.applied_fixes)

    assert "TRANSPILED_TO_BASIS" in result.applied_fixes
    assert "h" not in result.circuit.count_ops()
    assert set(result.circuit.count_ops()) <= {"u3"}
    assert Statevector.from_instruction(result.circuit).equiv(
        Statevector.from_instruction(circuit)
    )


def test_repair_strips_final_measurements_for_statevector_mode():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.measure_all()

    with pytest.raises(QiskitError):
        Statevector.from_instruction(circuit)

    result = repair_quantum_circuit(circuit, mode="statevector")

    _print_repair_case(
        "状态向量模式移除末尾测量",
        circuit,
        result.circuit,
        result.applied_fixes,
    )

    assert "STRIPPED_FINAL_MEASUREMENTS" in result.applied_fixes
    assert "measure" not in result.circuit.count_ops()
    assert result.circuit.num_clbits == 0
    np.testing.assert_allclose(
        Statevector.from_instruction(result.circuit).probabilities(),
        np.array([0.5, 0.5]),
    )


def test_repair_adds_measurements_for_sampling_mode():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    assert "measure" not in circuit.count_ops()

    result = repair_quantum_circuit(circuit, mode="sampling")

    _print_repair_case("抽样模式补充测量", circuit, result.circuit, result.applied_fixes)

    assert "ADDED_MEASUREMENTS" in result.applied_fixes
    assert result.circuit.num_clbits == 2
    assert result.circuit.count_ops()["measure"] == 2
