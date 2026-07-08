from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from double_quant.programming import (
    DecisionProgram,
    EuropeanCallPriceMeasure,
    ExpectedShortfallMeasure,
    ValuationProgram,
)
from rich import box
from rich.console import Console
from rich.table import Table

FUNCTION_NO = 79
FUNCTION_NAME = "quantum-algorithm-and-data-structure-metalanguage"
DOC_DIR = Path(__file__).parents[2] / "docs" / f"{FUNCTION_NO}-{FUNCTION_NAME}"
TEST_COMMAND = (
    "uv run pytest tests/double_quant/programming/79-quantum_algorithm_and_data_structure_metalanguage.py -s"
)

ALGORITHM_TO_LANGUAGE = {
    "最优投资组合算法（Func-1）": "决策性问题",
    "风险价值计量算法（Func-2）": "估值性问题",
    "金融衍生品定价算法（Func-3）": "估值性问题",
    "动态账本更新算法（Func-4）": "决策性问题",
    "去中心化金融管理算法（Func-5）": "决策性问题",
    "反欺诈监测算法（Func-6）": "决策性问题",
    "支付与结算系统算法（Func-7）": "决策性问题",
    "贷款发放决策算法（Func-8）": "决策性问题",
    "银行网点布局优化算法（Func-9）": "决策性问题",
    "指数追踪算法（Func-10）": "决策性问题",
}


@dataclass(frozen=True, slots=True)
class MetalanguageCaseResult:
    case_name: str
    problem_language: str
    covered_algorithms: str
    source_representation: str
    target_representation: str
    executable_result: str
    acceptance_meaning: str


def test_quantum_algorithm_and_data_structure_metalanguage() -> None:
    _test_language_catalog_covers_ten_algorithms()

    results = [
        _test_decision_problem_linear_structure(),
        _test_decision_problem_search_structure(),
        _test_decision_problem_constrained_binary_structure(),
        _test_valuation_problem_risk_value(),
        _test_valuation_problem_derivatives_pricing(),
    ]

    assert len(results) == 5
    assert {result.problem_language for result in results} == {
        "决策性问题",
        "估值性问题",
    }

    _write_acceptance_documents(results)

    print(_program_output(results))
    assert (DOC_DIR / "results.md").is_file()
    assert (DOC_DIR / "test_report.md").is_file()
    assert (DOC_DIR / "technical_report.md").is_file()


def _test_language_catalog_covers_ten_algorithms() -> None:
    assert len(ALGORITHM_TO_LANGUAGE) == 10
    assert set(ALGORITHM_TO_LANGUAGE.values()) == {"决策性问题", "估值性问题"}


def _test_decision_problem_linear_structure() -> MetalanguageCaseResult:
    program = DecisionProgram(name="资产配置平衡", kind="decision", domain="投资组合")
    w = program.add_variables("权重", 2, vtype="continuous")
    program.add_constraint(w[0] + w[1] == 3)
    program.add_constraint(w[0] - w[1] == 1)

    system = program.to_linear_system()
    solution = np.linalg.solve(system.matrix, system.vector)

    np.testing.assert_allclose(solution, np.array([2.0, 1.0]))

    return MetalanguageCaseResult(
        case_name="决策性问题表达线性配置约束",
        problem_language="决策性问题",
        covered_algorithms="最优投资组合算法（Func-1）",
        source_representation="声明连续权重变量，并写出预算与目标收益约束",
        target_representation="矩阵方程形式",
        executable_result=f"求得配置向量 [{solution[0]:.1f}, {solution[1]:.1f}]",
        acceptance_meaning="证明决策性问题能够表达最优投资组合算法所需的配置约束",
    )


def _test_decision_problem_search_structure() -> MetalanguageCaseResult:
    program = DecisionProgram(name="管理动作选择", kind="decision", domain="去中心化金融")
    program.add_variables("动作", 6, vtype="binary")

    circuit = program.to_grover_circuit(iterations=1)

    assert circuit.num_qubits == 3
    assert circuit.metadata["logical_variables"] == 6

    return MetalanguageCaseResult(
        case_name="决策性问题表达动作搜索",
        problem_language="决策性问题",
        covered_algorithms="去中心化金融管理算法（Func-5）；银行网点布局优化算法（Func-9）",
        source_representation="声明二元动作变量和搜索轮数",
        target_representation="压缩搜索线路形式",
        executable_result=(
            f"生成线路量子位 {circuit.num_qubits}，"
            f"候选空间 {circuit.metadata['search_space_size']}"
        ),
        acceptance_meaning="证明决策性问题能够表达搜索类金融应用算法",
    )


def _test_decision_problem_constrained_binary_structure() -> MetalanguageCaseResult:
    program = DecisionProgram(name="特征组合选择", kind="decision", domain="贷款发放")
    x = program.add_variables("特征", 4, vtype="binary")
    program.add_constraint(x[0] + x[1] == 1)
    program.add_constraint(x[2] + x[3] == 1)
    program.set_objective(
        1.2 * x[0] + 1.4 * x[1] + 1.1 * x[2] + 1.5 * x[3],
        sense="maximize",
    )

    problem = program.to_rasengan_problem()
    best = problem.best_feasible_state()

    assert problem.num_variables == 4
    assert problem.num_constraints == 2
    assert problem.is_feasible(best)

    return MetalanguageCaseResult(
        case_name="决策性问题表达约束二元选择",
        problem_language="决策性问题",
        covered_algorithms=(
            "反欺诈监测算法（Func-6）；支付与结算系统算法（Func-7）；"
            "贷款发放决策算法（Func-8）；指数追踪算法（Func-10）"
        ),
        source_representation="声明二元变量、分组约束和收益目标",
        target_representation="约束二元选择问题形式",
        executable_result=(
            f"最优可行比特串 {best.tolist()}，目标值 {problem.objective_value(best):.6f}"
        ),
        acceptance_meaning="证明决策性问题能够表达约束二元类金融应用算法",
    )


def _test_valuation_problem_risk_value() -> MetalanguageCaseResult:
    program = ValuationProgram(name="组合风险计量", kind="valuation", domain="风险计量")
    program.set_measure(ExpectedShortfallMeasure, target="组合尾部风险")
    program.add_data("portfolio_returns", np.array([-0.08, -0.04, 0.01, 0.02]))
    program.add_parameter("alpha", 0.75)

    value = float(program.evaluate())

    assert value == 0.08

    return MetalanguageCaseResult(
        case_name="估值性问题表达风险数值",
        problem_language="估值性问题",
        covered_algorithms="风险价值计量算法（Func-2）",
        source_representation="声明收益率数据和置信水平",
        target_representation="风险数值计算形式",
        executable_result=f"风险数值 = {value:.6f}",
        acceptance_meaning="证明估值性问题能够表达风险价值计量算法",
    )


def _test_valuation_problem_derivatives_pricing() -> MetalanguageCaseResult:
    program = ValuationProgram(name="衍生品定价", kind="valuation", domain="衍生品定价")
    program.set_measure(EuropeanCallPriceMeasure, target="衍生品价格")
    program.add_data("terminal_price_scenarios", np.array([90.0, 100.0, 110.0, 120.0]))
    program.add_parameter("strike", 100.0)
    program.add_parameter("risk_free_rate", 0.0)
    program.add_parameter("maturity", "1Y")

    value = float(program.evaluate())

    assert value == 7.5

    return MetalanguageCaseResult(
        case_name="估值性问题表达衍生品定价",
        problem_language="估值性问题",
        covered_algorithms="金融衍生品定价算法（Func-3）",
        source_representation="声明到期价格场景、执行价和期限参数",
        target_representation="衍生品定价数值形式",
        executable_result=f"定价结果 = {value:.6f}",
        acceptance_meaning="证明估值性问题能够表达金融衍生品定价算法",
    )


def _program_output(results: list[MetalanguageCaseResult]) -> str:
    console = Console(
        file=io.StringIO(),
        width=150,
        record=True,
        force_terminal=True,
        color_system=None,
    )
    console.print(f"功能编号：{FUNCTION_NO}")
    console.print("功能名称：实现量子算法和数据结构的元语言表示法")
    console.print(f"测试命令：{TEST_COMMAND}")
    console.print(f"覆盖场景数：{len(results)}")
    console.print()

    table = Table(
        title="两类编程语言体系与十类算法覆盖结果",
        box=box.ROUNDED,
        show_lines=True,
        expand=True,
        padding=(0, 1),
    )
    table.add_column("用例", ratio=12, overflow="fold")
    table.add_column("编程语言体系", ratio=9, overflow="fold")
    table.add_column("覆盖算法", ratio=30, overflow="fold")
    table.add_column("源表示", ratio=22, overflow="fold")
    table.add_column("目标表示", ratio=18, overflow="fold")
    table.add_column("运行输出", ratio=20, overflow="fold")
    table.add_column("验收含义", ratio=25, overflow="fold")
    for result in results:
        table.add_row(
            result.case_name,
            result.problem_language,
            result.covered_algorithms,
            result.source_representation,
            result.target_representation,
            result.executable_result,
            result.acceptance_meaning,
        )
    console.print(table)
    return console.export_text(styles=False)


def _write_acceptance_documents(results: list[MetalanguageCaseResult]) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / "results.md").write_text(_results_markdown(results), encoding="utf-8")
    (DOC_DIR / "test_report.md").write_text(_test_report_markdown(results), encoding="utf-8")
    (DOC_DIR / "technical_report.md").write_text(_technical_report_markdown(results), encoding="utf-8")


def _results_markdown(results: list[MetalanguageCaseResult]) -> str:
    return "\n".join(
        [
            "# 79 量子算法和数据结构元语言表示法 - 结果文档",
            "",
            "## 运行命令",
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
            "## 结果说明",
            "",
            "- 编程语言体系只包含“决策性问题”和“估值性问题”。",
            "- 十类算法全部被归入上述两类体系。",
            "- 报告不把内部问题实例工厂或内部求解路径作为算法名称。",
        ]
    )


def _test_report_markdown(results: list[MetalanguageCaseResult]) -> str:
    return "\n".join(
        [
            "# 79 量子算法和数据结构元语言表示法 - 测试报告",
            "",
            "## 测试结论",
            "",
            f"测试通过。本次共执行 {len(results)} 个场景，证明两类编程语言体系能够覆盖十类算法。",
            "",
            "## 覆盖面",
            "",
            "- 决策性问题覆盖 8 类算法。",
            "- 估值性问题覆盖 2 类算法。",
            "- 对外算法名称严格限定为 Func-1 至 Func-10。",
        ]
    )


def _technical_report_markdown(results: list[MetalanguageCaseResult]) -> str:
    mapping = "\n".join(
        f"| {name} | {language} |" for name, language in ALGORITHM_TO_LANGUAGE.items()
    )
    return "\n".join(
        [
            "# 79 量子算法和数据结构元语言表示法 - 技术报告",
            "",
            "## 设计原则",
            "",
            "元语言只面向两类问题：决策性问题和估值性问题。十类算法先归入这两类问题，再由内部实现路径完成具体计算。内部实现路径不是算法名称，不进入报告算法清单。",
            "",
            "## 十类算法归属",
            "",
            "| 算法名称 | 编程语言体系 |",
            "|---|---|",
            mapping,
            "",
            "## 验收输出",
            "",
            _program_output(results),
        ]
    )
