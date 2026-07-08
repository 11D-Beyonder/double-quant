from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from double_quant.application import (
    AntifraudMonitoringAlgorithm,
    BranchLocationAlgorithm,
    DefiManagementAlgorithm,
    DynamicLedgerUpdateAlgorithm,
    IndexTrackingAlgorithm,
    LoanDecisionAlgorithm,
    PaymentSettlementAlgorithm,
)
from double_quant.programming import default_operator_library
from rich import box
from rich.console import Console
from rich.table import Table

FUNCTION_NO = 88
FUNCTION_NAME = "quantum-researcher-algorithm-configuration-interface"
DOC_DIR = Path(__file__).parents[2] / "docs" / f"{FUNCTION_NO}-{FUNCTION_NAME}"
TEST_COMMAND = (
    "uv run pytest tests/double_quant/programming/88-quantum_researcher_algorithm_configuration_interface.py -s"
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
class ResearcherCaseResult:
    algorithm_name: str
    problem_language: str
    configurable_point: str
    actual_output: str
    validation_summary: str


def test_quantum_researcher_algorithm_configuration_interface() -> None:
    results = [
        _test_portfolio_configuration(),
        _test_risk_value_configuration(),
        _test_derivatives_pricing_configuration(),
        _test_dynamic_ledger_configuration(),
        _test_defi_configuration(),
        _test_antifraud_configuration(),
        _test_payment_configuration(),
        _test_loan_configuration(),
        _test_branch_configuration(),
        _test_index_tracking_configuration(),
    ]

    assert len(results) == 10
    assert {result.algorithm_name for result in results} == set(ALGORITHM_NAMES.values())
    assert {result.problem_language for result in results} == {"决策性问题", "估值性问题"}

    _write_acceptance_documents(results)

    print(_program_output(results))
    assert (DOC_DIR / "results.md").is_file()
    assert (DOC_DIR / "test_report.md").is_file()
    assert (DOC_DIR / "technical_report.md").is_file()


def _test_portfolio_configuration() -> ResearcherCaseResult:
    library = default_operator_library()
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
    weights = result.financial_result["weights"]
    assert set(weights) == {"资产甲", "资产乙"}
    return ResearcherCaseResult(
        algorithm_name=ALGORITHM_NAMES["func_1"],
        problem_language=PROBLEM_LANGUAGE["func_1"],
        configurable_point="预期收益、协方差矩阵、目标收益、资产名称",
        actual_output=f"资产甲 {weights['资产甲']:.6f}，资产乙 {weights['资产乙']:.6f}",
        validation_summary="输出权重包含两项资产",
    )


def _test_risk_value_configuration() -> ResearcherCaseResult:
    library = default_operator_library()
    result = library.execute(
        "func_2",
        {"portfolio_returns": np.array([0.01, -0.03, 0.02, -0.08]), "alpha": 0.75},
    )
    value = result.financial_result["expected_shortfall"]
    assert value == pytest.approx(0.08)
    return ResearcherCaseResult(
        algorithm_name=ALGORITHM_NAMES["func_2"],
        problem_language=PROBLEM_LANGUAGE["func_2"],
        configurable_point="收益率序列、置信水平",
        actual_output=f"风险数值 {value:.6f}",
        validation_summary="风险数值与预期一致",
    )


def _test_derivatives_pricing_configuration() -> ResearcherCaseResult:
    library = default_operator_library()
    result = library.execute(
        "func_3",
        {
            "terminal_price_scenarios": np.array([90.0, 100.0, 110.0, 120.0]),
            "strike": 100.0,
            "risk_free_rate": 0.0,
            "maturity": "1Y",
        },
    )
    value = result.financial_result["option_price"]
    assert value == pytest.approx(7.5)
    return ResearcherCaseResult(
        algorithm_name=ALGORITHM_NAMES["func_3"],
        problem_language=PROBLEM_LANGUAGE["func_3"],
        configurable_point="到期价格场景、执行价、无风险利率、期限",
        actual_output=f"定价结果 {value:.6f}",
        validation_summary="定价结果与预期一致",
    )


def _test_dynamic_ledger_configuration() -> ResearcherCaseResult:
    circuit = DynamicLedgerUpdateAlgorithm(modulus=15, base=2, phase_qubits=6).build_circuit()
    assert circuit.metadata["application_id"] == "Func-4"
    return _circuit_case(
        "func_4",
        "账本模数、底数、相位寄存器规模",
        circuit.num_qubits,
        circuit.depth(),
    )


def _test_defi_configuration() -> ResearcherCaseResult:
    circuit = DefiManagementAlgorithm(logical_variables=6, grover_iterations=1).build_circuit()
    assert circuit.metadata["application_id"] == "Func-5"
    return _circuit_case("func_5", "管理动作数量、迭代轮数", circuit.num_qubits, circuit.depth())


def _test_antifraud_configuration() -> ResearcherCaseResult:
    circuit = AntifraudMonitoringAlgorithm(groups=2, layers=1).build_circuit()
    assert circuit.metadata["application_id"] == "Func-6"
    return _circuit_case("func_6", "交易闭环组数、线路层数", circuit.num_qubits, circuit.depth())


def _test_payment_configuration() -> ResearcherCaseResult:
    circuit = PaymentSettlementAlgorithm(accounts=3, layers=1).build_circuit()
    assert circuit.metadata["application_id"] == "Func-7"
    return _circuit_case("func_7", "账户数量、线路层数", circuit.num_qubits, circuit.depth())


def _test_loan_configuration() -> ResearcherCaseResult:
    circuit = LoanDecisionAlgorithm(feature_groups=3, layers=1).build_circuit()
    assert circuit.metadata["application_id"] == "Func-8"
    return _circuit_case("func_8", "特征组数量、线路层数", circuit.num_qubits, circuit.depth())


def _test_branch_configuration() -> ResearcherCaseResult:
    circuit = BranchLocationAlgorithm(candidate_sites=8, grover_iterations=2).build_circuit()
    assert circuit.metadata["application_id"] == "Func-9"
    return _circuit_case("func_9", "候选网点数量、迭代轮数", circuit.num_qubits, circuit.depth())


def _test_index_tracking_configuration() -> ResearcherCaseResult:
    circuit = IndexTrackingAlgorithm(sectors=3, layers=1).build_circuit()
    assert circuit.metadata["application_id"] == "Func-10"
    return _circuit_case("func_10", "行业分组数量、线路层数", circuit.num_qubits, circuit.depth())


def _circuit_case(
    operator_id: str,
    configurable_point: str,
    num_qubits: int,
    depth: int,
) -> ResearcherCaseResult:
    return ResearcherCaseResult(
        algorithm_name=ALGORITHM_NAMES[operator_id],
        problem_language=PROBLEM_LANGUAGE[operator_id],
        configurable_point=configurable_point,
        actual_output=f"量子位 {num_qubits}，线路深度 {depth}",
        validation_summary="功能编号标记正确，线路可构造",
    )


def _write_acceptance_documents(results: list[ResearcherCaseResult]) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / "results.md").write_text(_results_document(results), encoding="utf-8")
    (DOC_DIR / "test_report.md").write_text(_test_report_document(results), encoding="utf-8")
    (DOC_DIR / "technical_report.md").write_text(
        _technical_report_document(results), encoding="utf-8"
    )


def _program_output(results: list[ResearcherCaseResult]) -> str:
    console = Console(
        file=io.StringIO(),
        width=150,
        record=True,
        force_terminal=True,
        color_system=None,
    )
    console.print(f"功能编号：{FUNCTION_NO}")
    console.print("功能名称：为量子研究人员提供量子算法、配置接口")
    console.print(f"测试命令：{TEST_COMMAND}")
    console.print(f"覆盖用例数：{len(results)}")
    console.print()

    table = Table(
        title="十类算法配置接口运行结果",
        box=box.ROUNDED,
        show_lines=True,
        expand=True,
        padding=(0, 1),
    )
    table.add_column("算法名称", ratio=24, overflow="fold")
    table.add_column("编程语言体系", ratio=10, overflow="fold")
    table.add_column("可配置项", ratio=28, overflow="fold")
    table.add_column("实际输出", ratio=22, overflow="fold")
    table.add_column("断言校验", ratio=24, overflow="fold")
    for result in results:
        table.add_row(
            result.algorithm_name,
            result.problem_language,
            result.configurable_point,
            result.actual_output,
            result.validation_summary,
        )
    console.print(table)
    return console.export_text(styles=False)


def _results_document(results: list[ResearcherCaseResult]) -> str:
    return "\n".join(
        [
            "# 88 为量子研究人员提供量子算法、配置接口 - 结果文档",
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
            "- 配置接口只展示十类算法名称。",
            "- 编程语言体系只展示“决策性问题”和“估值性问题”。",
            "- 内部线路构造方式不作为额外算法名称展示。",
        ]
    )


def _test_report_document(results: list[ResearcherCaseResult]) -> str:
    return "\n".join(
        [
            "# 88 为量子研究人员提供量子算法、配置接口 - 测试报告",
            "",
            "## 测试结论",
            "",
            f"测试通过。本次共覆盖 {len(results)} 个配置用例，对应 Func-1 至 Func-10 十类算法。",
            "",
            "## 覆盖面分析",
            "",
            "- 决策性问题配置项覆盖 8 类算法。",
            "- 估值性问题配置项覆盖 2 类算法。",
            "- 报告中未引入十类算法之外的算法名称。",
        ]
    )


def _technical_report_document(results: list[ResearcherCaseResult]) -> str:
    return "\n".join(
        [
            "# 88 为量子研究人员提供量子算法、配置接口 - 技术报告",
            "",
            "## 技术目标",
            "",
            "本功能提供十类算法的参数化配置入口。对外只呈现算法名称、编程语言体系、可配置项和运行结果；内部实现细节不作为算法名称进入报告。",
            "",
            "## 验收输出",
            "",
            _program_output(results),
        ]
    )
