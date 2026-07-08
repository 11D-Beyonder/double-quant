from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from double_quant.algorithm.hhl import HHLSolver
from double_quant.algorithm.qubo import NumPyMinimumEigensolverSolver
from double_quant.algorithm.shapley import BinaryEnumerationCalculator
from double_quant.application.risk import RiskAttributor
from double_quant.common.metric import expected_shortfall
from double_quant.programming import (
    DecisionProgram,
    ExpectedShortfallMeasure,
    ShapleyRiskContributionMeasure,
    ValuationProgram,
)
from rich import box
from rich.console import Console
from rich.table import Table

FUNCTION_NO = 87
FUNCTION_NAME = "financial-scenario-programming-interface"
DOC_DIR = Path(__file__).parents[2] / "docs" / f"{FUNCTION_NO}-{FUNCTION_NAME}"
TEST_COMMAND = (
    "uv run pytest tests/double_quant/programming/87-financial_scenario_programming_interface.py -s"
)


@dataclass(frozen=True, slots=True)
class InterfaceCaseResult:
    case_name: str
    interface_type: str
    business_goal: str
    key_outputs: str
    lowered_threshold: str


def test_financial_scenario_programming_interface_end_to_end():
    results = [
        _test_decision_program_solves_qubo_end_to_end(),
        _test_decision_program_solves_hhl_linear_system_end_to_end(),
        _test_valuation_program_evaluates_expected_shortfall_end_to_end(),
        _test_valuation_program_solves_shapley_risk_contribution_end_to_end(),
    ]

    assert [result.case_name for result in results] == [
        "单点QUBO资产选择",
        "单点HHL线性系统",
        "单点预期损失估值",
        "单点风险贡献归因",
    ]
    assert {result.interface_type for result in results} == {"decision", "valuation"}

    _write_acceptance_documents(results)

    print(_program_output(results))
    assert (DOC_DIR / "results.md").is_file()
    assert (DOC_DIR / "test_report.md").is_file()
    assert (DOC_DIR / "technical_report.md").is_file()


def _test_decision_program_solves_qubo_end_to_end() -> InterfaceCaseResult:
    program = DecisionProgram(
        name="single_point_qubo",
        kind="decision",
        domain="portfolio",
    )
    program.add_data("assets", ["asset_a", "asset_b"])
    program.add_output("selected_assets")
    x = program.add_variables("x", 2, vtype="binary")
    program.set_objective(
        -1.0 * x[0] - 2.0 * x[1] + 4.0 * x[0] * x[1],
        sense="minimize",
    )

    qubo = program.to_qubo_problem()
    result = NumPyMinimumEigensolverSolver().solve(qubo)

    assert qubo.variable_names == ["x_0", "x_1"]
    assert qubo.evaluate([0, 1]) == pytest.approx(-2.0)
    assert result.best_bitstring.tolist() == [0, 1]
    assert result.best_objective == pytest.approx(-2.0)

    return InterfaceCaseResult(
        case_name="单点QUBO资产选择",
        interface_type="decision",
        business_goal="组合选择/离散决策",
        key_outputs="QUBO 变量名 x_0,x_1；最优比特串 [0,1]；最优目标值 -2.0",
        lowered_threshold="金融人员只需声明变量和目标函数，无需手写 QUBO 转换与求解细节",
    )


def _test_decision_program_solves_hhl_linear_system_end_to_end() -> InterfaceCaseResult:
    program = DecisionProgram(
        name="single_point_linear_system",
        kind="decision",
        domain="portfolio",
    )
    program.add_output("weights")
    x = program.add_variables("x", 2)
    program.add_constraints(
        [
            x[0] + x[1] == 3.0,
            x[0] - x[1] == 1.0,
        ]
    )

    system = program.to_linear_system()
    hhl_solution = HHLSolver.solve(
        system.matrix,
        system.vector,
        "sapo",
        max_qpe_qubits=4,
    )

    np.testing.assert_allclose(system.matrix, np.array([[1.0, 1.0], [1.0, -1.0]]))
    np.testing.assert_allclose(system.vector, np.array([3.0, 1.0]))
    np.testing.assert_allclose(hhl_solution, np.array([2.0, 1.0]), atol=1.0e-8)

    return InterfaceCaseResult(
        case_name="单点HHL线性系统",
        interface_type="decision",
        business_goal="约束权重求解/线性系统建模",
        key_outputs="线性系统矩阵 [[1,1],[1,-1]]；右端向量 [3,1]；解向量 [2,1]",
        lowered_threshold="金融人员只需写等式约束，无需手动组装 Ax=b 或直接操作 HHL 细节",
    )


def _test_valuation_program_evaluates_expected_shortfall_end_to_end() -> InterfaceCaseResult:
    returns = np.array([0.01, -0.03, 0.02, -0.08, -0.04, 0.03])
    program = ValuationProgram(
        name="single_point_expected_shortfall",
        kind="valuation",
        domain="risk",
    )
    program.add_data("portfolio_returns", returns)
    program.add_parameter("alpha", 0.75)
    program.set_measure(ExpectedShortfallMeasure, target="portfolio")
    program.add_output("expected_shortfall")

    actual = program.evaluate()
    expected = expected_shortfall(returns, 0.75)

    assert actual == pytest.approx(expected)

    return InterfaceCaseResult(
        case_name="单点预期损失估值",
        interface_type="valuation",
        business_goal="组合尾部风险估值",
        key_outputs=f"预期损失(alpha=0.75) = {actual:.6f}",
        lowered_threshold="金融人员只需传收益率和 alpha 参数，无需重复实现风险度量公式",
    )


def _test_valuation_program_solves_shapley_risk_contribution_end_to_end() -> InterfaceCaseResult:
    returns = pd.DataFrame(
        {
            "asset_a": [0.01, -0.02, 0.03, -0.04, 0.02, -0.01],
            "asset_b": [-0.01, 0.02, -0.03, 0.01, -0.05, 0.03],
        }
    )
    program = ValuationProgram(
        name="single_point_risk_contribution",
        kind="valuation",
        domain="risk_attribution",
    )
    program.add_data("asset_returns", returns)
    program.add_parameter("alpha", 0.75)
    program.add_parameter("mode", "es")
    program.add_parameter("solver_class", BinaryEnumerationCalculator)
    program.set_measure(
        ShapleyRiskContributionMeasure,
        target="portfolio",
        breakdown="asset",
    )
    program.add_output("risk_contribution")

    actual = program.evaluate()
    expected = RiskAttributor(
        returns,
        BinaryEnumerationCalculator,
        alpha=0.75,
        mode="es",
    ).attribute()

    assert actual == pytest.approx(expected)

    summary = ", ".join(f"{name}={value:.6f}" for name, value in actual.items())
    return InterfaceCaseResult(
        case_name="单点风险贡献归因",
        interface_type="valuation",
        business_goal="资产级风险贡献拆解",
        key_outputs=f"Shapley 风险贡献：{summary}",
        lowered_threshold="金融人员只需声明度量与求解器类型，无需直接编排 Shapley 子集枚举流程",
    )


def _write_acceptance_documents(results: list[InterfaceCaseResult]) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / "results.md").write_text(_results_document(results), encoding="utf-8")
    (DOC_DIR / "test_report.md").write_text(_test_report_document(results), encoding="utf-8")
    (DOC_DIR / "technical_report.md").write_text(
        _technical_report_document(results), encoding="utf-8"
    )


def _program_output(results: list[InterfaceCaseResult]) -> str:
    console = Console(
        file=io.StringIO(),
        width=150,
        record=True,
        force_terminal=True,
        color_system=None,
    )
    console.print(f"功能编号：{FUNCTION_NO}")
    console.print("功能名称：为金融人员提供金融场景的操作、算法接口")
    console.print(f"测试命令：{TEST_COMMAND}")
    console.print(f"覆盖场景数：{len(results)}")
    console.print()

    table = Table(
        title="金融场景接口运行结果",
        box=box.ROUNDED,
        show_lines=True,
        expand=True,
        padding=(0, 1),
    )
    table.add_column("用例", ratio=15, overflow="fold")
    table.add_column("接口类型", ratio=10, overflow="fold")
    table.add_column("金融场景目标", ratio=20, overflow="fold")
    table.add_column("关键输出", ratio=32, overflow="fold")
    table.add_column("降低门槛说明", ratio=38, overflow="fold")
    for result in results:
        table.add_row(
            result.case_name,
            result.interface_type,
            result.business_goal,
            result.key_outputs,
            result.lowered_threshold,
        )
    console.print(table)
    return console.export_text(styles=False)


def _results_document(results: list[InterfaceCaseResult]) -> str:
    return "\n".join(
        [
            "# 87 面向金融人员的金融场景操作与算法接口测试（Func-54）结果文档",
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
            "- 已验证 `DecisionProgram` 可以承载金融人员常见的离散决策建模与线性约束建模。",
            "- 已验证 `ValuationProgram` 可以承载金融人员常见的尾部风险估值与 Shapley 风险归因。",
            "- 已验证编程接口将金融描述层与底层算法层隔离，金融人员不需要直接操作 QUBO、HHL 和 Shapley 内部细节。",
            "- 三份验收文档由测试脚本自动写入，满足交付要求。",
        ]
    )


def _test_report_document(results: list[InterfaceCaseResult]) -> str:
    interface_types = "、".join(sorted({result.interface_type for result in results}))
    return "\n".join(
        [
            "# 87 面向金融人员的金融场景操作与算法接口测试（Func-54）测试报告",
            "",
            "## 测试结论",
            "",
            f"测试通过。本次共覆盖 {len(results)} 个金融场景接口用例，覆盖接口类型：{interface_types}。结果表明，当前编程框架已经能够以声明式方式向金融人员暴露金融场景操作与算法接口。",
            "",
            "## 测试命令",
            "",
            f"`{TEST_COMMAND}`",
            "",
            "## 覆盖场景分析",
            "",
            "- 组合选择场景：验证金融人员只通过变量、目标函数声明即可进入 QUBO 求解路径。",
            "- 权重求解场景：验证金融人员只通过线性约束即可进入 HHL 线性系统求解路径。",
            "- 风险度量场景：验证金融人员只通过收益率数组与 alpha 参数即可计算预期损失。",
            "- 风险归因场景：验证金融人员只通过资产收益表、度量类型和求解器类型即可完成 Shapley 风险贡献分解。",
            "",
            "## 对降低开发门槛的验证",
            "",
            "- 接口入口统一：金融人员面向 `DecisionProgram` / `ValuationProgram` 工作，而不是分别学习多个底层求解器 API。",
            "- 领域语义清晰：`domain`、`measure`、`solver_class`、`output` 等字段保持金融语义，不暴露过多量子实现细节。",
            "- 组合能力明确：同一框架同时容纳决策类问题和估值类问题，减少跨模块心智切换成本。",
            "",
            "## 风险与限制",
            "",
            "- 当前测试聚焦接口可用性与结果正确性，不评估大规模市场数据下的性能。",
            "- HHL 路径依赖对称/Hermitian 线性系统输入，复杂投资组合约束仍需调用方保证可转换性。",
            "- 风险归因示例使用 `mode=\"es\"` 的经典路径；若走量子路径，应遵守仓库中关于超可加值函数的约束。",
        ]
    )


def _technical_report_document(results: list[InterfaceCaseResult]) -> str:
    cases = "、".join(result.case_name for result in results)
    return "\n".join(
        [
            "# 87 面向金融人员的金融场景操作与算法接口测试（Func-54）技术报告",
            "",
            "## 技术目标",
            "",
            "本功能的目标是将底层量子金融算法封装为更贴近金融人员认知的声明式编程接口，使其能够以“定义数据、变量、约束、度量和输出”的方式完成金融场景建模，而不必直接处理量子算法调用细节。",
            "",
            "## 接口设计概览",
            "",
            "当前接口设计采用“编程框架层 -> 金融问题层 -> 算法层”的三层结构：",
            "",
            "1. 编程框架层：`double_quant.programming` 暴露统一入口。",
            "2. 金融问题层：`DecisionProgram` 表达决策问题，`ValuationProgram` 表达估值与风险分析问题。",
            "3. 算法层：由框架内部路由到底层 `QUBO`、`HHLSolver`、`RiskAttributor` 与 Shapley 求解器。",
            "",
            "## 核心接口设计",
            "",
            "### 1. `DecisionProgram`",
            "",
            "`DecisionProgram` 面向金融人员的决策建模场景，核心设计点如下：",
            "",
            "- 用 `add_variables()` 声明投资决策变量，支持连续、二值、整数等变量类型。",
            "- 用 `set_objective()` 声明目标函数，可自然表达收益最大化、成本最小化、风险惩罚等形式。",
            "- 用 `add_constraint()` / `add_constraints()` 声明业务约束，如预算约束、持仓约束、风险暴露约束。",
            "- 通过 `to_qubo_problem()` 将离散决策自动下沉为 QUBO 表示。",
            "- 通过 `to_linear_system()` 将线性约束自动整理为 `Ax=b` 形式，以接入 HHL 求解链路。",
            "",
            "该设计的关键意义在于：金融人员写的是金融规则，而不是底层数学格式转换代码。",
            "",
            "### 2. `ValuationProgram`",
            "",
            "`ValuationProgram` 面向金融估值、风险分析与归因场景，核心设计点如下：",
            "",
            "- 用 `add_data()` 注入收益率、价格路径、资产收益矩阵等金融数据。",
            "- 用 `add_parameter()` 注入 `alpha`、`strike`、`solver_class`、`mode` 等计算参数。",
            "- 用 `set_measure()` 绑定可导入的度量类，例如 `ExpectedShortfallMeasure` 和 `ShapleyRiskContributionMeasure`。",
            "- 通过 `evaluate()` 统一触发估值过程，并在执行前自动检查所需数据和参数是否齐全。",
            "",
            "该设计的关键意义在于：金融人员面向“度量”工作，而不是面向某个特定算法实现工作。",
            "",
            "## 算法接口如何被封装",
            "",
            "### 1. QUBO 接口封装",
            "",
            "在 `single_point_qubo` 用例中，金融人员只定义资产选择变量和目标函数。编程框架自动把表达式树转换为 QUBO，再由 `NumPyMinimumEigensolverSolver` 求解。用户无需自己维护变量顺序、QUBO 系数矩阵和最优比特串反解。",
            "",
            "### 2. HHL 接口封装",
            "",
            "在 `single_point_linear_system` 用例中，金融人员只写两条线性等式约束。编程框架自动整理出矩阵和右端向量，然后调用 `HHLSolver.solve()`。这说明金融场景里的权重平衡或资金分配问题可以通过声明式方式进入量子线性求解流程。",
            "",
            "### 3. 风险度量接口封装",
            "",
            "在 `single_point_expected_shortfall` 用例中，金融人员只需要提供收益率数组与 `alpha` 参数，并选择 `ExpectedShortfallMeasure`。框架屏蔽了尾部样本筛选与均值计算的实现细节。",
            "",
            "### 4. 风险归因接口封装",
            "",
            "在 `single_point_risk_contribution` 用例中，金融人员只需要提供资产收益矩阵、置信水平和求解器类型，框架便通过 `ShapleyRiskContributionMeasure` 调用 `RiskAttributor` 完成资产级风险分解。",
            "",
            "## 本次测试验证的接口能力",
            "",
            f"本次测试覆盖的 4 个用例分别为：{cases}。它们共同验证了以下能力：",
            "",
            "- 同一编程框架同时支持决策类与估值类金融场景。",
            "- 金融语义接口可以稳定映射到底层量子/经典算法模块。",
            "- 接口输出结果具备可验证性，可与底层基准实现直接对照。",
            "- 接口抽象能够明显降低金融人员的开发进入门槛。",
            "",
            "## 结论",
            "",
            "该功能已经具备“为金融人员提供金融场景的操作、算法接口”的核心特征：上层接口保持金融领域表达，下层自动衔接量子金融算法与经典基线算法。测试结果表明，该接口设计既保证了可用性，也为后续扩展更多金融模板、更多度量类型和更多量子求解后端预留了统一入口。",
        ]
    )
