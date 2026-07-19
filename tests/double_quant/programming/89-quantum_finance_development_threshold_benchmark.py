from __future__ import annotations

import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from double_quant.algorithm.rasengan import build_rasengan_circuit
from double_quant.application import DefiManagementAlgorithm, DynamicLedgerUpdateAlgorithm
from double_quant.common.metric import expected_shortfall
from double_quant.programming import (
    DecisionProgram,
    EuropeanCallPriceMeasure,
    ExpectedShortfallMeasure,
    ValuationProgram,
    default_operator_library,
)

SKILL_SCRIPT_DIR = Path(__file__).parents[3] / ".codex" / "skills" / "3rd-test" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPT_DIR))
from chinese_plot_style import (  # type: ignore[reportMissingImports]  # noqa: E402
    DOUBLE_COLUMN_MM,
    apply_chinese_style,
    save_figure,
    style_axes,
)

FUNCTION_NO = 89
FUNCTION_NAME = "quantum-finance-development-threshold-benchmark"
TESTS_ROOT = Path(__file__).parents[2]
PROJECT_ROOT = TESTS_ROOT.parent
DOC_DIR = TESTS_ROOT / "docs" / f"{FUNCTION_NO}-{FUNCTION_NAME}"
TEST_COMMAND = (
    "uv run pytest tests/double_quant/programming/89-quantum_finance_development_threshold_benchmark.py -s"
)


@dataclass(frozen=True, slots=True)
class BenchmarkCaseResult:
    algorithm_name: str
    problem_language: str
    file_stem: str
    high_level_loc: int
    low_level_loc: int
    reduced_loc: int
    reduction_ratio: float
    output_summary: str
    threshold_value: str
    high_level_code: str
    low_level_code: str
    high_level_file: str
    low_level_file: str


def test_quantum_finance_development_threshold_benchmark() -> None:
    results = [
        _benchmark_risk_value_case(),
        _benchmark_derivatives_pricing_case(),
        _benchmark_portfolio_case(),
        _benchmark_dynamic_ledger_case(),
        _benchmark_defi_case(),
        _benchmark_loan_case(),
    ]

    assert len(results) == 6
    assert all(result.low_level_loc > result.high_level_loc for result in results)
    assert all(result.reduced_loc > 0 for result in results)
    assert all(result.reduction_ratio > 0 for result in results)

    _write_acceptance_documents(results)

    print(_program_output(results))
    assert (DOC_DIR / "results.md").is_file()
    assert (DOC_DIR / "test_report.md").is_file()
    assert (DOC_DIR / "technical_report.md").is_file()


def _benchmark_risk_value_case() -> BenchmarkCaseResult:
    returns = np.array([0.01, -0.03, 0.02, -0.08, -0.04, 0.03])
    alpha = 0.75

    high_level_code = '''
program = ValuationProgram(name="组合风险计量", kind="valuation", domain="风险计量")
program.add_data("portfolio_returns", returns)
program.add_parameter("alpha", alpha)
program.set_measure(ExpectedShortfallMeasure, target="组合尾部风险")
value = program.evaluate()
'''
    low_level_code = '''
returns_array = np.asarray(returns, dtype=float)
if returns_array.ndim != 1 or returns_array.size == 0:
    raise ValueError("收益率序列必须是一维非空数组")
if not 0.0 < alpha < 1.0:
    raise ValueError("置信水平必须在 0 到 1 之间")
sorted_returns = np.sort(returns_array)
tail_count = max(1, int(np.ceil((1.0 - alpha) * sorted_returns.size)))
tail_losses = sorted_returns[:tail_count]
value = -float(np.mean(tail_losses))
'''

    program = ValuationProgram(name="组合风险计量", kind="valuation", domain="风险计量")
    program.add_data("portfolio_returns", returns)
    program.add_parameter("alpha", alpha)
    program.set_measure(ExpectedShortfallMeasure, target="组合尾部风险")
    actual = program.evaluate()
    expected = expected_shortfall(returns, alpha)
    assert actual == expected

    return _make_result(
        algorithm_name="风险价值计量算法（Func-2）",
        problem_language="估值性问题",
        file_stem="01_风险价值计量算法",
        high_level_code=high_level_code,
        low_level_code=low_level_code,
        output_summary=f"风险数值={actual:.6f}",
        threshold_value="使用者只声明收益率数据和置信水平，不需要手写尾部样本筛选流程",
    )


def _benchmark_derivatives_pricing_case() -> BenchmarkCaseResult:
    scenarios = np.array([90.0, 100.0, 110.0, 120.0])

    high_level_code = '''
program = ValuationProgram(name="衍生品定价", kind="valuation", domain="衍生品定价")
program.add_data("terminal_price_scenarios", scenarios)
program.add_parameter("strike", 100.0)
program.add_parameter("risk_free_rate", 0.0)
program.add_parameter("maturity", "1Y")
program.set_measure(EuropeanCallPriceMeasure, target="衍生品价格")
value = program.evaluate()
'''
    low_level_code = '''
scenario_array = np.asarray(scenarios, dtype=float)
if scenario_array.ndim != 1 or scenario_array.size == 0:
    raise ValueError("到期价格场景必须是一维非空数组")
strike = 100.0
risk_free_rate = 0.0
maturity = 1.0
payoff = np.maximum(scenario_array - strike, 0.0)
discount = np.exp(-risk_free_rate * maturity)
value = float(discount * np.mean(payoff))
'''

    program = ValuationProgram(name="衍生品定价", kind="valuation", domain="衍生品定价")
    program.add_data("terminal_price_scenarios", scenarios)
    program.add_parameter("strike", 100.0)
    program.add_parameter("risk_free_rate", 0.0)
    program.add_parameter("maturity", "1Y")
    program.set_measure(EuropeanCallPriceMeasure, target="衍生品价格")
    value = program.evaluate()
    assert value == 7.5

    return _make_result(
        algorithm_name="金融衍生品定价算法（Func-3）",
        problem_language="估值性问题",
        file_stem="02_金融衍生品定价算法",
        high_level_code=high_level_code,
        low_level_code=low_level_code,
        output_summary=f"定价结果={value:.6f}",
        threshold_value="使用者只声明价格场景和合约参数，不需要手写收益函数和贴现聚合",
    )


def _benchmark_portfolio_case() -> BenchmarkCaseResult:
    high_level_code = '''
library = default_operator_library()
result = library.execute(
    "func_1",
    {
        "expected_returns": expected_returns,
        "covariance": covariance,
        "target_return": target_return,
        "assets": ["资产甲", "资产乙"],
    },
    max_qpe_qubits=4,
)
weights = result.financial_result["weights"]
'''
    low_level_code = '''
expected_returns = np.asarray(expected_returns, dtype=float)
covariance = np.asarray(covariance, dtype=float)
matrix = np.zeros((4, 4), dtype=float)
matrix[0, 2:] = expected_returns
matrix[1, 2:] = 1.0
matrix[2:, 0] = expected_returns
matrix[2:, 1] = 1.0
matrix[2:, 2:] = covariance
vector = np.zeros(4, dtype=float)
vector[0] = target_return
vector[1] = 1.0
solution = np.linalg.solve(matrix, vector)
weights = {"资产甲": float(solution[2]), "资产乙": float(solution[3])}
'''

    expected_returns = np.array([0.02, 0.03])
    covariance = np.array([[0.1, 0.02], [0.02, 0.12]])
    target_return = 0.025
    result = default_operator_library().execute(
        "func_1",
        {
            "expected_returns": expected_returns,
            "covariance": covariance,
            "target_return": target_return,
            "assets": ["资产甲", "资产乙"],
        },
        max_qpe_qubits=4,
    )
    weights = result.financial_result["weights"]
    assert set(weights) == {"资产甲", "资产乙"}

    return _make_result(
        algorithm_name="最优投资组合算法（Func-1）",
        problem_language="决策性问题",
        file_stem="03_最优投资组合算法",
        high_level_code=high_level_code,
        low_level_code=low_level_code,
        output_summary=f"输出资产甲={weights['资产甲']:.6f}，资产乙={weights['资产乙']:.6f}",
        threshold_value="使用者只声明收益、协方差和目标收益，不需要手写约束矩阵组装",
    )


def _benchmark_dynamic_ledger_case() -> BenchmarkCaseResult:
    high_level_code = '''
algorithm = DynamicLedgerUpdateAlgorithm(modulus=15, base=2, phase_qubits=6)
circuit = algorithm.build_circuit()
'''
    low_level_code = '''
phase_qubits = 6
work_qubits = 4
circuit = QuantumCircuit(phase_qubits + work_qubits, phase_qubits)
phase_register = list(range(phase_qubits))
work_register = list(range(phase_qubits, phase_qubits + work_qubits))
circuit.h(phase_register)
circuit.x(work_register[0])
for target in work_register[1:]:
    circuit.cswap(phase_register[0], work_register[0], target)
circuit.cswap(phase_register[1], work_register[0], work_register[2])
circuit.cswap(phase_register[1], work_register[1], work_register[3])
for index in range(phase_qubits // 2):
    circuit.swap(phase_register[index], phase_register[phase_qubits - index - 1])
for target_index in range(phase_qubits):
    for control_index in range(target_index):
        angle = -math.pi / float(2 ** (target_index - control_index))
        circuit.cp(angle, phase_register[control_index], phase_register[target_index])
    circuit.h(phase_register[target_index])
circuit.measure(phase_register, phase_register)
'''

    circuit = DynamicLedgerUpdateAlgorithm(modulus=15, base=2, phase_qubits=6).build_circuit()
    assert circuit.metadata["application_id"] == "Func-4"

    return _make_result(
        algorithm_name="动态账本更新算法（Func-4）",
        problem_language="决策性问题",
        file_stem="04_动态账本更新算法",
        high_level_code=high_level_code,
        low_level_code=low_level_code,
        output_summary=f"量子位={circuit.num_qubits}，线路深度={circuit.depth()}",
        threshold_value="使用者只配置账本模数和寄存器规模，不需要手写账本更新线路细节",
    )


def _benchmark_defi_case() -> BenchmarkCaseResult:
    high_level_code = '''
algorithm = DefiManagementAlgorithm(logical_variables=8, grover_iterations=2)
circuit = algorithm.build_circuit()
'''
    low_level_code = '''
logical_variables = 8
num_qubits = math.ceil(logical_variables / 2)
circuit = QuantumCircuit(num_qubits, num_qubits)
circuit.h(range(num_qubits))
for _ in range(2):
    target = num_qubits - 1
    circuit.h(target)
    circuit.mcx(list(range(target)), target)
    circuit.h(target)
    circuit.h(range(num_qubits))
    circuit.x(range(num_qubits))
    circuit.h(target)
    circuit.mcx(list(range(target)), target)
    circuit.h(target)
    circuit.x(range(num_qubits))
    circuit.h(range(num_qubits))
circuit.measure(range(num_qubits), range(num_qubits)[::-1])
'''

    circuit = DefiManagementAlgorithm(logical_variables=8, grover_iterations=2).build_circuit()
    assert circuit.metadata["application_id"] == "Func-5"

    return _make_result(
        algorithm_name="去中心化金融管理算法（Func-5）",
        problem_language="决策性问题",
        file_stem="05_去中心化金融管理算法",
        high_level_code=high_level_code,
        low_level_code=low_level_code,
        output_summary=f"量子位={circuit.num_qubits}，线路深度={circuit.depth()}",
        threshold_value="使用者只声明管理动作规模和迭代轮数，不需要手写搜索线路",
    )


def _benchmark_loan_case() -> BenchmarkCaseResult:
    high_level_code = '''
program = DecisionProgram(name="贷款特征选择", kind="decision", domain="贷款发放")
x = program.add_variables("特征", 4, vtype="binary")
program.add_constraints([x[0] + x[1] == 1, x[2] + x[3] == 1])
program.set_objective(
    1.0 * x[0] + 1.2 * x[1] + 0.9 * x[2] + 1.1 * x[3],
    sense="maximize",
)
problem = program.to_rasengan_problem()
circuit = build_rasengan_circuit(problem, layers=1)
best = problem.best_feasible_state()
'''
    low_level_code = '''
linear = np.array([1.0, 1.2, 0.9, 1.1], dtype=float)
constraints = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
rhs = np.array([1.0, 1.0], dtype=float)
variable_names = ("特征_0", "特征_1", "特征_2", "特征_3")
problem = LinearConstraintBinaryProblem(
    linear=linear,
    constraints=constraints,
    rhs=rhs,
    sense="max",
    variable_names=variable_names,
)
transition_basis = np.array([[-1, 1, 0, 0], [0, 0, -1, 1]], dtype=int)
feasible_state = np.array([1, 0, 1, 0], dtype=int)
circuit = build_rasengan_circuit(
    problem,
    layers=1,
    transition_basis=transition_basis,
    feasible_state=feasible_state,
)
best = problem.best_feasible_state()
'''

    program = DecisionProgram(name="贷款特征选择", kind="decision", domain="贷款发放")
    x = program.add_variables("特征", 4, vtype="binary")
    program.add_constraints([x[0] + x[1] == 1, x[2] + x[3] == 1])
    program.set_objective(
        1.0 * x[0] + 1.2 * x[1] + 0.9 * x[2] + 1.1 * x[3],
        sense="maximize",
    )
    problem = program.to_rasengan_problem()
    circuit = build_rasengan_circuit(problem, layers=1)
    best = problem.best_feasible_state()
    assert problem.is_feasible(best)

    return _make_result(
        algorithm_name="贷款发放决策算法（Func-8）",
        problem_language="决策性问题",
        file_stem="06_贷款发放决策算法",
        high_level_code=high_level_code,
        low_level_code=low_level_code,
        output_summary=f"最优可行比特串={best.tolist()}，线路量子位={circuit.num_qubits}",
        threshold_value="使用者只写变量、约束和目标，不需要手写约束矩阵和初始可行态",
    )


def _make_result(
    *,
    algorithm_name: str,
    problem_language: str,
    file_stem: str,
    high_level_code: str,
    low_level_code: str,
    output_summary: str,
    threshold_value: str,
) -> BenchmarkCaseResult:
    high_level_code = textwrap.dedent(high_level_code).strip()
    low_level_code = textwrap.dedent(low_level_code).strip()
    high_level_loc = _count_effective_loc(high_level_code)
    low_level_loc = _count_effective_loc(low_level_code)
    reduced_loc = low_level_loc - high_level_loc
    reduction_ratio = reduced_loc / low_level_loc
    return BenchmarkCaseResult(
        algorithm_name=algorithm_name,
        problem_language=problem_language,
        file_stem=file_stem,
        high_level_loc=high_level_loc,
        low_level_loc=low_level_loc,
        reduced_loc=reduced_loc,
        reduction_ratio=reduction_ratio,
        output_summary=output_summary,
        threshold_value=threshold_value,
        high_level_code=high_level_code,
        low_level_code=low_level_code,
        high_level_file=f"code_examples/{file_stem}_高层接口.py",
        low_level_file=f"code_examples/{file_stem}_原始实现.py",
    )


def _count_effective_loc(code: str) -> int:
    return sum(1 for line in code.splitlines() if line.strip() and not line.lstrip().startswith("#"))


def _program_output(results: list[BenchmarkCaseResult]) -> str:
    average_high = sum(result.high_level_loc for result in results) / len(results)
    average_low = sum(result.low_level_loc for result in results) / len(results)
    average_reduced = sum(result.reduced_loc for result in results) / len(results)
    average_ratio = sum(result.reduction_ratio for result in results) / len(results)
    lines = [
        f"功能编号：{FUNCTION_NO}",
        "功能名称：量子金融软件开发门槛降低测试",
        f"测试命令：{TEST_COMMAND}",
        f"覆盖场景数：{len(results)}",
        "",
        "开发门槛降低指标表",
        "| 算法名称 | 编程语言体系 | 原始实现行数 | 高层接口行数 | 降低行数 | 代码行数降低比例 | 结果校验 | 门槛降低体现 |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for result in results:
        lines.append(
            "| "
            f"{result.algorithm_name} | "
            f"{result.problem_language} | "
            f"{result.low_level_loc} | "
            f"{result.high_level_loc} | "
            f"{result.reduced_loc} | "
            f"{result.reduction_ratio:.1%} | "
            f"{result.output_summary} | "
            f"{result.threshold_value} |"
        )
    lines.extend(
        [
            "",
            "汇总指标",
            f"- 平均原始实现行数：{average_low:.1f} 行",
            f"- 平均高层接口行数：{average_high:.1f} 行",
            f"- 平均降低行数：{average_reduced:.1f} 行",
            f"- 平均代码行数降低比例：{average_ratio:.1%}",
        ]
    )
    return "\n".join(lines)


def _write_acceptance_documents(results: list[BenchmarkCaseResult]) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    _write_code_example_files(results)
    _write_threshold_reduction_chart(results)
    (DOC_DIR / "results.md").write_text(_results_document(results), encoding="utf-8")
    (DOC_DIR / "test_report.md").write_text(_test_report_document(results), encoding="utf-8")
    (DOC_DIR / "technical_report.md").write_text(_technical_report_document(results), encoding="utf-8")


def _write_code_example_files(results: list[BenchmarkCaseResult]) -> None:
    examples_dir = DOC_DIR / "code_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    for old_file in examples_dir.glob("*.py"):
        old_file.unlink()

    for result in results:
        (DOC_DIR / result.low_level_file).write_text(
            f"# ruff: noqa: F821\n{result.low_level_code}\n",
            encoding="utf-8",
        )
        (DOC_DIR / result.high_level_file).write_text(
            f"# ruff: noqa: F821\n{result.high_level_code}\n",
            encoding="utf-8",
        )


def _write_threshold_reduction_chart(results: list[BenchmarkCaseResult]) -> None:
    images_dir = DOC_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    required_text = "开发门槛降低指标原始实现高层接口代码行数降低比例"
    figsize = apply_chinese_style(
        width_mm=DOUBLE_COLUMN_MM,
        ncols=2,
        nrows=1,
        panel_aspect=1.15,
        required_text=required_text,
    )
    plt.rcParams["savefig.pad_inches"] = 0.18
    fig, (line_ax, ratio_ax) = plt.subplots(1, 2, figsize=figsize)

    labels = [_chart_label(result.algorithm_name) for result in results]
    y_positions = np.arange(len(results))
    low_values = np.asarray([result.low_level_loc for result in results], dtype=float)
    high_values = np.asarray([result.high_level_loc for result in results], dtype=float)
    ratios = np.asarray([result.reduction_ratio * 100.0 for result in results], dtype=float)

    line_ax.barh(
        y_positions - 0.18,
        low_values,
        height=0.34,
        label="原始实现行数",
        color="#64748b",
    )
    line_ax.barh(
        y_positions + 0.18,
        high_values,
        height=0.34,
        label="高层接口行数",
        color="#16a34a",
    )
    for index, (low_value, high_value) in enumerate(zip(low_values, high_values, strict=True)):
        line_ax.text(low_value + 0.4, index - 0.18, f"{low_value:.0f}", va="center")
        line_ax.text(high_value + 0.4, index + 0.18, f"{high_value:.0f}", va="center")
    line_ax.set_yticks(y_positions, labels)
    line_ax.invert_yaxis()
    line_ax.legend(loc="upper right")
    style_axes(
        line_ax,
        title="原始实现与高层接口行数对比",
        xlabel="有效代码行数",
        ylabel="算法场景",
    )

    colors = ["#0f766e" if ratio >= 50.0 else "#2563eb" for ratio in ratios]
    ratio_ax.barh(y_positions, ratios, height=0.5, color=colors)
    for index, ratio in enumerate(ratios):
        ratio_ax.text(ratio + 1.0, index, f"降低 {ratio:.1f}%", va="center")
    ratio_ax.set_xlim(0.0, max(100.0, float(ratios.max()) + 8.0))
    ratio_ax.set_yticks(y_positions, labels)
    ratio_ax.invert_yaxis()
    style_axes(
        ratio_ax,
        title="代码行数降低比例",
        xlabel="降低比例（%）",
        ylabel="",
    )

    average_ratio = sum(result.reduction_ratio for result in results) / len(results)
    average_reduced = sum(result.reduced_loc for result in results) / len(results)
    fig.suptitle("开发门槛降低指标图", y=0.98, fontsize=11, fontweight="semibold")
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.94), w_pad=2.8)
    fig.text(
        0.5,
        0.045,
        f"平均代码行数降低比例 {average_ratio:.1%}，平均减少 {average_reduced:.1f} 行",
        ha="center",
        va="center",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#ecfdf5", "edgecolor": "#16a34a"},
    )
    save_figure(fig, images_dir, "development_threshold_reduction", formats=("png",))
    plt.close(fig)


def _chart_label(algorithm_name: str) -> str:
    return (
        algorithm_name.replace("算法（", "\n（")
        .replace("管理", "管理")
        .replace("最优投资组合", "最优投资组合")
    )


def _results_document(results: list[BenchmarkCaseResult]) -> str:
    average_ratio = sum(result.reduction_ratio for result in results) / len(results)
    strongest = max(results, key=lambda item: item.reduction_ratio)
    metrics_table = _metrics_markdown_table(results)
    return "\n".join(
        [
            "# 89 降低量子金融软件开发门槛 - 结果文档",
            "",
            "## 测试命令",
            "",
            "```bash",
            TEST_COMMAND,
            "```",
            "",
            "## 程序输出",
            "",
            "```text",
            _program_output(results),
            "```",
            "",
            "## 图片输出",
            "",
            "![终端运行截图](images/terminal_run.png)",
            "",
            "![开发门槛降低指标图](images/development_threshold_reduction.png)",
            "",
            "## 关键指标",
            "",
            metrics_table,
            "",
            "## 总体结论",
            "",
            f"- {len(results)} 个代表场景的平均代码行数降低比例为 {average_ratio:.1%}。",
            f"- 代码行数降低比例最高的场景是“{strongest.algorithm_name}”，降低比例为 {strongest.reduction_ratio:.1%}。",
            "- 结果文档只展示十类算法名称和两类编程语言体系。",
            "- 代码示例文件已输出到本目录的 `code_examples` 子目录。",
        ]
    )


def _test_report_document(results: list[BenchmarkCaseResult]) -> str:
    average_high = sum(result.high_level_loc for result in results) / len(results)
    average_low = sum(result.low_level_loc for result in results) / len(results)
    average_reduced = sum(result.reduced_loc for result in results) / len(results)
    average_ratio = sum(result.reduction_ratio for result in results) / len(results)
    return "\n".join(
        [
            "# 89 降低量子金融软件开发门槛 - 测试报告",
            "",
            "## 测试结论",
            "",
            f"测试通过。本次覆盖 {len(results)} 个代表场景，平均原始实现 {average_low:.1f} 行，高层接口 {average_high:.1f} 行，平均减少 {average_reduced:.1f} 行，平均代码行数降低比例 {average_ratio:.1%}。",
            "",
            "## 指标证据",
            "",
            "- `images/development_threshold_reduction.png` 展示原始实现行数、高层接口行数和代码行数降低比例。",
            "- `images/terminal_run.png` 展示完整字段的终端运行结果，不使用省略号表达关键指标。",
            "",
            "## 口径说明",
            "",
            "- 算法名称只使用 Func-1 至 Func-10 中的名称。",
            "- 编程语言体系只使用“决策性问题”和“估值性问题”。",
            "- 内部实现路径不作为算法名称写入报告。",
        ]
    )


def _technical_report_document(results: list[BenchmarkCaseResult]) -> str:
    return "\n".join(
        [
            "# 89 降低量子金融软件开发门槛 - 技术报告",
            "",
            "## 技术目标",
            "",
            "本测试用代码行数对比量化高层编程接口对开发门槛的降低效果。高层接口面向十类算法和两类编程语言体系，屏蔽内部线路构造、矩阵组装、约束展开和数值后处理细节。",
            "",
            "## 指标图实现",
            "",
            "脚本生成 `images/development_threshold_reduction.png`，左侧对比原始实现与高层接口的有效代码行数，右侧展示每个算法场景的代码行数降低比例。",
            "",
            "## 验收输出",
            "",
            _program_output(results),
        ]
    )


def _metrics_markdown_table(results: list[BenchmarkCaseResult]) -> str:
    lines = [
        "| 算法名称 | 编程语言体系 | 原始实现行数 | 高层接口行数 | 降低行数 | 代码行数降低比例 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| "
            f"{result.algorithm_name} | "
            f"{result.problem_language} | "
            f"{result.low_level_loc} | "
            f"{result.high_level_loc} | "
            f"{result.reduced_loc} | "
            f"{result.reduction_ratio:.1%} |"
        )
    return "\n".join(lines)
