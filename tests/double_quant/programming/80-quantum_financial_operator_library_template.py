from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from double_quant.programming import (
    OperatorResult,
    TemplateResult,
    default_operator_library,
    default_software_templates,
)
from rich import box
from rich.console import Console
from rich.table import Table

FUNCTION_NO = 80
FUNCTION_NAME = "quantum-financial-operator-library-template"
DOC_DIR = Path(__file__).parents[2] / "docs" / f"{FUNCTION_NO}-{FUNCTION_NAME}"
TEST_COMMAND = (
    "uv run pytest "
    "tests/double_quant/programming/80-quantum_financial_operator_library_template.py -s"
)

ALGORITHM_NAMES = {
    "func_1": "最优投资组合算法（Func-1）",
    "func_2": "风险价值计量算法（Func-2）",
    "func_3": "金融衍生品定价算法（Func-3）",
    "func_4": "动态账本更新算法（Func-4）",
    "func_5": "去中心化金融管理算法（Func-5）",
    "func_6": "反欺诈监测算法（Func-6）",
    "func_7": "支付与结算系统算法（Func-7）",
    "func_8": "贷款发放决策算法（Func-8）",
    "func_9": "银行网点布局优化算法（Func-9）",
    "func_10": "指数追踪算法（Func-10）",
}

PROBLEM_LANGUAGE = {
    "func_1": "决策性问题",
    "func_2": "估值性问题",
    "func_3": "估值性问题",
    "func_4": "决策性问题",
    "func_5": "决策性问题",
    "func_6": "决策性问题",
    "func_7": "决策性问题",
    "func_8": "决策性问题",
    "func_9": "决策性问题",
    "func_10": "决策性问题",
}


@dataclass(frozen=True, slots=True)
class OperatorCaseResult:
    case_name: str
    problem_language: str
    algorithm_name: str
    execution_summary: str
    acceptance_meaning: str


def test_quantum_financial_operator_library_and_software_template() -> None:
    library = default_operator_library()
    templates = default_software_templates()
    specs_by_id = {spec.id: spec for spec in library.list_specs()}

    assert set(specs_by_id) == set(ALGORITHM_NAMES)
    assert [specs_by_id[key].name for key in _ordered_algorithm_ids()] == [
        ALGORITHM_NAMES[key] for key in _ordered_algorithm_ids()
    ]
    _assert_algorithm_library_catalog(specs_by_id)

    results = [
        _test_portfolio_algorithm(library),
        _test_risk_value_algorithm(library),
        _test_derivatives_pricing_algorithm(library),
        *_test_application_algorithms(library),
        _test_application_template_covers_declared_algorithms(library, templates),
    ]

    assert len(results) == 11
    assert {result.problem_language for result in results} == {"决策性问题", "估值性问题"}

    _write_acceptance_documents(results)

    print(_program_output(results))
    assert (DOC_DIR / "results.md").is_file()
    assert (DOC_DIR / "test_report.md").is_file()
    assert (DOC_DIR / "technical_report.md").is_file()


def _ordered_algorithm_ids() -> list[str]:
    return [f"func_{index}" for index in range(1, 11)]


def _assert_algorithm_library_catalog(specs_by_id) -> None:
    names = [specs_by_id[key].name for key in _ordered_algorithm_ids()]
    assert names == [ALGORITHM_NAMES[key] for key in _ordered_algorithm_ids()]


def _test_portfolio_algorithm(library) -> OperatorCaseResult:
    result = library.execute(
        "func_1",
        {
            "expected_returns": np.array([0.02, 0.03]),
            "covariance": np.array([[0.1, 0.02], [0.02, 0.12]]),
            "target_return": 0.025,
            "assets": ["资产甲", "资产乙"],
        },
        max_qpe_qubits=4,
    )

    assert isinstance(result, OperatorResult)
    assert result.operator_id == "func_1"
    assert set(result.financial_result["weights"]) == {"资产甲", "资产乙"}

    weights = result.financial_result["weights"]
    return OperatorCaseResult(
        case_name="执行最优投资组合算法",
        problem_language=PROBLEM_LANGUAGE["func_1"],
        algorithm_name=ALGORITHM_NAMES["func_1"],
        execution_summary=(
            f"输出资产甲权重 {weights['资产甲']:.6f}，资产乙权重 {weights['资产乙']:.6f}"
        ),
        acceptance_meaning="证明决策性问题能够进入最优投资组合算法并输出资产权重",
    )


def _test_risk_value_algorithm(library) -> OperatorCaseResult:
    result = library.execute(
        "func_2",
        {"portfolio_returns": np.array([0.01, -0.03, 0.02, -0.08]), "alpha": 0.75},
    )

    assert isinstance(result, OperatorResult)
    assert result.operator_id == "func_2"
    assert result.financial_result["expected_shortfall"] == pytest.approx(0.08)

    return OperatorCaseResult(
        case_name="执行风险价值计量算法",
        problem_language=PROBLEM_LANGUAGE["func_2"],
        algorithm_name=ALGORITHM_NAMES["func_2"],
        execution_summary="尾部风险数值为 0.080000",
        acceptance_meaning="证明估值性问题能够进入风险价值计量算法并输出风险数值",
    )


def _test_derivatives_pricing_algorithm(library) -> OperatorCaseResult:
    result = library.execute(
        "func_3",
        {
            "terminal_price_scenarios": np.array([90.0, 100.0, 110.0, 120.0]),
            "strike": 100.0,
            "risk_free_rate": 0.0,
            "maturity": "1Y",
        },
    )

    assert isinstance(result, OperatorResult)
    assert result.operator_id == "func_3"
    assert result.financial_result["option_price"] == pytest.approx(7.5)

    return OperatorCaseResult(
        case_name="执行金融衍生品定价算法",
        problem_language=PROBLEM_LANGUAGE["func_3"],
        algorithm_name=ALGORITHM_NAMES["func_3"],
        execution_summary="衍生品定价结果为 7.500000",
        acceptance_meaning="证明估值性问题能够进入金融衍生品定价算法并输出定价结果",
    )


def _test_application_algorithms(library) -> list[OperatorCaseResult]:
    results: list[OperatorCaseResult] = []
    for operator_id in [f"func_{index}" for index in range(4, 11)]:
        result = library.execute(operator_id, {}, backend="application_circuit")
        assert isinstance(result, OperatorResult)
        assert result.operator_id == operator_id
        assert result.resources.num_qubits is not None
        assert result.resources.circuit_depth is not None
        results.append(
            OperatorCaseResult(
                case_name=f"执行{ALGORITHM_NAMES[operator_id]}",
                problem_language=PROBLEM_LANGUAGE[operator_id],
                algorithm_name=ALGORITHM_NAMES[operator_id],
                execution_summary=(
                    f"构造完成，量子位 {result.resources.num_qubits}，"
                    f"线路深度 {result.resources.circuit_depth}"
                ),
                acceptance_meaning=(
                    f"证明决策性问题能够进入{ALGORITHM_NAMES[operator_id]}并形成可执行线路"
                ),
            )
        )
    return results


def _test_application_template_covers_declared_algorithms(
    library,
    templates,
) -> OperatorCaseResult:
    template = templates["template.func_application_catalog"]
    result = template.run(library, {})

    assert isinstance(result, TemplateResult)
    assert result.template_id == "template.func_application_catalog"
    assert result.operator_ids == tuple(f"func_{index}" for index in range(4, 11))
    assert len(result.step_results) == 7

    summary = "；".join(
        f"{ALGORITHM_NAMES[step.operator_id]} {step.resources.num_qubits} 量子位"
        for step in result.step_results
    )
    return OperatorCaseResult(
        case_name="应用算法模板覆盖核验",
        problem_language="决策性问题",
        algorithm_name="；".join(ALGORITHM_NAMES[f"func_{index}"] for index in range(4, 11)),
        execution_summary=summary,
        acceptance_meaning="证明软件模板只编排 Func-4 至 Func-10 七个应用算法，没有引入其他算法名称",
    )


def _write_acceptance_documents(results: list[OperatorCaseResult]) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / "results.md").write_text(_results_document(results), encoding="utf-8")
    (DOC_DIR / "test_report.md").write_text(_test_report_document(results), encoding="utf-8")
    (DOC_DIR / "technical_report.md").write_text(
        _technical_report_document(results), encoding="utf-8"
    )


def _program_output(results: list[OperatorCaseResult]) -> str:
    console = Console(
        file=io.StringIO(),
        width=150,
        record=True,
        force_terminal=True,
        color_system=None,
    )
    console.print(f"功能编号：{FUNCTION_NO}")
    console.print("功能名称：设计量子金融算子库与软件模板为主体的编程框架")
    console.print(f"测试命令：{TEST_COMMAND}")
    console.print(f"覆盖用例数：{len(results)}")
    console.print()

    table = Table(
        title="十类算法库与软件模板运行结果",
        box=box.ROUNDED,
        show_lines=True,
        expand=True,
        padding=(0, 1),
    )
    table.add_column("用例", ratio=14, overflow="fold")
    table.add_column("编程语言体系", ratio=10, overflow="fold")
    table.add_column("算法名称", ratio=30, overflow="fold")
    table.add_column("运行结果", ratio=26, overflow="fold")
    table.add_column("验收含义", ratio=30, overflow="fold")
    for result in results:
        table.add_row(
            result.case_name,
            result.problem_language,
            result.algorithm_name,
            result.execution_summary,
            result.acceptance_meaning,
        )
    console.print(table)
    return console.export_text(styles=False)


def _results_document(results: list[OperatorCaseResult]) -> str:
    return "\n".join(
        [
            "# 80 量子金融算子库与软件模板 - 结果文档",
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
            "- 算法库只登记 Func-1 至 Func-10 十类算法。",
            "- 编程语言体系只分为“决策性问题”和“估值性问题”。",
            "- 软件模板只覆盖动态账本更新算法、去中心化金融管理算法、反欺诈监测算法、支付与结算系统算法、贷款发放决策算法、银行网点布局优化算法、指数追踪算法。",
            "- 内部问题实例工厂不作为算法展示，也不进入对外清单。",
        ]
    )


def _test_report_document(results: list[OperatorCaseResult]) -> str:
    return "\n".join(
        [
            "# 80 量子金融算子库与软件模板 - 测试报告",
            "",
            "## 测试结论",
            "",
            f"测试通过。本次共执行 {len(results)} 个功能用例，算法名称严格限定为 Func-1 至 Func-10 十类算法。",
            "",
            "## 覆盖面分析",
            "",
            "- 决策性问题覆盖：最优投资组合算法、动态账本更新算法、去中心化金融管理算法、反欺诈监测算法、支付与结算系统算法、贷款发放决策算法、银行网点布局优化算法、指数追踪算法。",
            "- 估值性问题覆盖：风险价值计量算法、金融衍生品定价算法。",
            "- 对外结果表中没有出现内部工厂名或内部求解路径名。",
            "",
            "## 结论",
            "",
            "量子金融算子库与软件模板已经按指定十类算法完成统一登记和运行核验，满足算法名称统一要求。",
        ]
    )


def _technical_report_document(results: list[OperatorCaseResult]) -> str:
    algorithms = "\n".join(
        f"| {index} | {ALGORITHM_NAMES[f'func_{index}']} | {PROBLEM_LANGUAGE[f'func_{index}']} |"
        for index in range(1, 11)
    )
    return "\n".join(
        [
            "# 80 量子金融算子库与软件模板 - 技术报告",
            "",
            "## 技术实现",
            "",
            "本功能将对外算法清单限定为十类算法，并把编程语言体系限定为两类问题：决策性问题与估值性问题。底层线路构造、矩阵求解、搜索和约束处理都只作为内部实现路径，不作为额外算法名称展示。",
            "",
            "## 十类算法与编程语言体系",
            "",
            "| 序号 | 算法名称 | 编程语言体系 |",
            "|---|---|---|",
            algorithms,
            "",
            "## 软件模板",
            "",
            "软件模板面向七个应用型决策算法批量构造线路，包括 Func-4 至 Func-10。模板输出只使用十类算法名称，不暴露内部问题实例工厂。",
            "",
            "## 验收输出",
            "",
            _program_output(results),
        ]
    )
