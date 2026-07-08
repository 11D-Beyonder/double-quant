from pathlib import Path
import importlib.util
import sys
import types

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from simulator.analysis import compare_complexity


OUTPUT_IMAGE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "143-circuit_depth_optimization"
    / "images"
    / "143_depth_optimization.png"
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "double_quant" / "algorithm"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_algorithm_builders():
    grover = _load_module("dq_grover_circuit", SRC_ROOT / "grover" / "circuit.py")
    shor = _load_module("dq_shor_circuit", SRC_ROOT / "shor" / "circuit.py")
    hhl = _load_module("dq_hhl_variants", SRC_ROOT / "hhl" / "variants.py")
    rasengan_model = _load_module(
        "double_quant.algorithm.rasengan.model",
        SRC_ROOT / "rasengan" / "model.py",
    )
    _load_module(
        "double_quant.algorithm.rasengan.linear_system",
        SRC_ROOT / "rasengan" / "linear_system.py",
    )
    rasengan = _load_module("dq_rasengan_circuit", SRC_ROOT / "rasengan" / "circuit.py")
    shapley = _load_shapley_quantum_module()
    return grover, shor, hhl, rasengan_model, rasengan, shapley


def _load_shapley_quantum_module():
    qiskit_algorithms = types.ModuleType("qiskit_algorithms")
    for name in (
        "AmplitudeEstimation",
        "EstimationProblem",
        "FasterAmplitudeEstimation",
        "IterativeAmplitudeEstimation",
        "MaximumLikelihoodAmplitudeEstimation",
    ):
        setattr(qiskit_algorithms, name, type(name, (), {}))
    sys.modules.setdefault("qiskit_algorithms", qiskit_algorithms)

    common_util = types.ModuleType("double_quant.common.util")

    def normalize(values, denominator="max"):
        denominator_value = np.max(values)
        return values / denominator_value, denominator_value

    common_util.normalize = normalize
    sys.modules.setdefault("double_quant.common.util", common_util)

    _load_module(
        "double_quant.algorithm.shapley.protocol",
        SRC_ROOT / "shapley" / "protocol.py",
    )
    _load_module(
        "double_quant.algorithm.shapley.calculator",
        SRC_ROOT / "shapley" / "calculator.py",
    )
    return _load_module("dq_shapley_quantum", SRC_ROOT / "shapley" / "quantum.py")


def _build_hhl_circuit(hhl_module):
    matrix = np.array([[1.0, 0.25], [0.25, 1.5]])
    vector = np.array([1.0, 0.5])
    strategy = hhl_module.EigenBasedStrategy(matrix, vector, max_qpe_qubits=3)
    circuit = strategy._construct_circuit(*strategy._pre_scaling()).decompose(reps=1)
    circuit.name = "hhl_linear_solver"
    return circuit


def _build_qubo_qaoa_circuit():
    terms = []
    num_qubits = 4
    for index in range(num_qubits):
        label = ["I"] * num_qubits
        label[index] = "Z"
        terms.append(("".join(label), 0.2 + 0.1 * index))
    for left in range(num_qubits):
        for right in range(left + 1, num_qubits):
            label = ["I"] * num_qubits
            label[left] = "Z"
            label[right] = "Z"
            terms.append(("".join(label), 0.25))
    ansatz = QAOAAnsatz(cost_operator=SparsePauliOp.from_list(terms), reps=2)
    parameter_values = {
        parameter: 0.2 + 0.03 * index
        for index, parameter in enumerate(sorted(ansatz.parameters, key=lambda item: item.name))
    }
    circuit = ansatz.assign_parameters(parameter_values, inplace=False).decompose(reps=3)
    circuit.data = [item for item in circuit.data if item.operation.name != "barrier"]
    circuit.name = "qubo_qaoa_ansatz"
    return circuit


def _build_shapley_circuit(shapley_module):
    control_qubits = 4
    circuit = QuantumCircuit(control_qubits + 1, control_qubits + 1, name="quantum_shapley_value_loader")
    circuit.compose(shapley_module.IntervalLoader(control_qubits), list(range(control_qubits)), inplace=True)
    circuit.compose(shapley_module.VertexRotator(control_qubits), list(range(control_qubits + 1)), inplace=True)
    circuit.compose(
        shapley_module.ValueLoader(np.linspace(0.1, 1.0, 2**control_qubits)),
        list(range(control_qubits + 1)),
        inplace=True,
    )
    circuit.measure(range(control_qubits + 1), range(control_qubits + 1))
    return circuit.decompose(reps=2)


def _algorithm_circuits():
    grover, shor, hhl, rasengan_model, rasengan, shapley = _load_algorithm_builders()
    problem = rasengan_model.LinearConstraintBinaryProblem(
        linear=np.array([1.0, 2.0, 3.0, 4.0]),
        constraints=np.array([[1.0, 1.0, 1.0, 1.0]]),
        rhs=np.array([2.0]),
        sense="min",
    )
    transition_basis = np.array([[-1, 0, 1, 0], [0, -1, 0, 1]], dtype=int)
    feasible_state = np.array([1, 1, 0, 0], dtype=int)
    return [
        ("HHL线性求解", _build_hhl_circuit(hhl)),
        ("QUBO-QAOA", _build_qubo_qaoa_circuit()),
        ("Quantum Shapley", _build_shapley_circuit(shapley)),
        (
            "Shor周期查找",
            shor.build_shor_period_finding_circuit(
                15,
                base=2,
                phase_qubits=4,
                work_qubits=4,
            ),
        ),
        (
            "SFS-Grover搜索",
            grover.build_sfs_grover_circuit(
                logical_variables=8,
                compressed_qubits=4,
                iterations=2,
            ),
        ),
        (
            "Rasengan约束优化",
            rasengan.build_rasengan_circuit(
                problem,
                layers=1,
                transition_basis=transition_basis,
                feasible_state=feasible_state,
            ),
        ),
    ]


def _configure_chinese_font() -> None:
    candidates = [
        str(REPO_ROOT / "tests" / "docs" / "assets" / "fonts" / "STHeiti-Medium.ttc"),
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            font_manager.fontManager.addfont(candidate)
            font_name = font_manager.FontProperties(fname=candidate).get_name()
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _plot_depth(rows: list[dict[str, int | str]]) -> None:
    _configure_chinese_font()
    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)

    names = [str(row["name"]) for row in rows]
    before = [int(row["before"]) for row in rows]
    after = [int(row["after"]) for row in rows]
    x = list(range(len(rows)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    ax.bar([index - width / 2 for index in x], before, width, label="优化前电路深度")
    ax.bar([index + width / 2 for index in x], after, width, label="优化后电路深度")
    ax.set_title("真实算法电路深度优化对比")
    ax.set_ylabel("电路深度")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=16, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for index, row in enumerate(rows):
        ax.text(index - width / 2, int(row["before"]), str(row["before"]), ha="center", va="bottom", fontsize=8)
        ax.text(index + width / 2, int(row["after"]), str(row["after"]), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_IMAGE)
    plt.close(fig)


def main() -> None:
    rows = []
    for name, circuit in _algorithm_circuits():
        comparison = compare_complexity(circuit)
        baseline = comparison[0]
        optimized = min(comparison.values(), key=lambda report: report.depth)
        assert optimized.depth <= baseline.depth
        rows.append(
            {
                "name": name,
                "before": baseline.depth,
                "after": optimized.depth,
                "best_level": int(optimized.optimization_level or 0),
            }
        )

    _plot_depth(rows)

    print("143 仿真工具支持算法复杂度优化（真实算法深度优化）: PASS")
    for row in rows:
        reduction = int(row["before"]) - int(row["after"])
        print(f"{row['name']}优化前电路深度：{row['before']}")
        print(f"{row['name']}优化后最小电路深度：{row['after']}")
        print(f"{row['name']}电路深度减少：{reduction}")
        print(f"{row['name']}最佳优化等级：{row['best_level']}")
    print(f"深度优化对比图：{OUTPUT_IMAGE.relative_to(Path(__file__).resolve().parents[1])}")


if __name__ == "__main__":
    main()
