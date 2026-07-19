from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from double_quant.programming import ConstraintExpression, DecisionProgram, Expression
from double_quant.programming.optimizer import optimize_decision_program
from rich import box
from rich.console import Console
from rich.table import Table

FUNCTION_NO = 81
FUNCTION_NAME = "code-self-optimization"
DOC_DIR = Path(__file__).parents[2] / "docs" / f"{FUNCTION_NO}-{FUNCTION_NAME}"
IMAGE_DIR = DOC_DIR / "images"
CHART_PATH = IMAGE_DIR / "ir_redundancy_reduction.png"
TEST_COMMAND = (
    "uv run pytest tests/double_quant/programming/81-code_self_optimization.py"
)

CASE_LABELS = {
    "objective_zero_terms": "目标函数零项消除",
    "constraint_zero_quadratic": "约束零二次项消除",
    "all_zero_objective": "空目标函数压缩",
    "mixed_program_redundancy": "混合程序冗余消除",
}
SCOPE_LABELS = {
    "objective": "目标函数",
    "constraint": "约束",
    "program": "完整程序",
    "all": "汇总",
}


@dataclass(frozen=True, slots=True)
class OptimizationCase:
    name: str
    scope: str
    program: DecisionProgram
    reason: str


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    name: str
    scope: str
    before_terms: int
    after_terms: int
    reduced_terms: int
    reduction_rate: float
    reason: str


def test_code_self_optimization_reduces_ir_redundancy():
    results = [_evaluate_case(case) for case in _optimization_cases()]

    assert len(results) == 4
    assert _total_before(results) == 20
    assert _total_after(results) == 8
    assert _total_reduced(results) == 12
    assert all(result.reduced_terms > 0 for result in results)

    _write_acceptance_documents(results)

    print(_program_output(results))
    assert (DOC_DIR / "results.md").is_file()
    assert (DOC_DIR / "test_report.md").is_file()
    assert (DOC_DIR / "technical_report.md").is_file()
    assert CHART_PATH.is_file()


def _optimization_cases() -> list[OptimizationCase]:
    return [
        OptimizationCase(
            name="objective_zero_terms",
            scope="objective",
            program=_objective_zero_terms_program(),
            reason="删除零线性项、零二次项和近零常数项",
        ),
        OptimizationCase(
            name="constraint_zero_quadratic",
            scope="constraint",
            program=_constraint_zero_quadratic_program(),
            reason="删除约束中的零二次项，使约束中间表示恢复为线性形式",
        ),
        OptimizationCase(
            name="all_zero_objective",
            scope="objective",
            program=_all_zero_objective_program(),
            reason="删除全零目标函数中的无效线性项和二次项",
        ),
        OptimizationCase(
            name="mixed_program_redundancy",
            scope="program",
            program=_mixed_program_redundancy_program(),
            reason="同时删除目标函数和多个约束中的零项/近零项",
        ),
    ]


def _objective_zero_terms_program() -> DecisionProgram:
    program = DecisionProgram(
        name="objective_zero_terms",
        kind="decision",
        domain="portfolio",
    )
    program.add_variables("x", 2, vtype="binary")
    program.set_objective(
        Expression(
            linear={"x_0": 0.0, "x_1": 2.0},
            quadratic={("x_0", "x_1"): 0.0, ("x_1", "x_1"): 3.0},
            constant=1.0e-13,
        ),
        sense="minimize",
    )
    return program


def _constraint_zero_quadratic_program() -> DecisionProgram:
    program = DecisionProgram(
        name="constraint_zero_quadratic",
        kind="decision",
        domain="portfolio",
    )
    program.add_variables("x", 1)
    program.add_constraint(
        ConstraintExpression(
            Expression(linear={"x_0": 1.0}, quadratic={("x_0", "x_0"): 0.0}),
            "==",
        )
    )
    return program


def _all_zero_objective_program() -> DecisionProgram:
    program = DecisionProgram(
        name="all_zero_objective",
        kind="decision",
        domain="portfolio",
    )
    program.add_variables("z", 2, vtype="binary")
    program.set_objective(
        Expression(
            linear={"z_0": 0.0, "z_1": 0.0},
            quadratic={("z_0", "z_1"): 0.0},
        ),
        sense="minimize",
    )
    return program


def _mixed_program_redundancy_program() -> DecisionProgram:
    program = DecisionProgram(
        name="mixed_program_redundancy",
        kind="decision",
        domain="portfolio",
    )
    program.add_variables("x", 3, vtype="binary")
    program.set_objective(
        Expression(
            linear={"x_0": 1.0, "x_1": 0.0, "x_2": 2.0},
            quadratic={("x_0", "x_1"): 0.0, ("x_1", "x_2"): 4.0},
        ),
        sense="minimize",
    )
    program.add_constraint(
        ConstraintExpression(
            Expression(linear={"x_0": 1.0, "x_1": 0.0}, constant=1.0e-13),
            "<=",
        )
    )
    program.add_constraint(
        ConstraintExpression(
            Expression(linear={"x_2": 3.0}, quadratic={("x_1", "x_1"): 0.0}),
            "==",
        )
    )
    return program


def _evaluate_case(case: OptimizationCase) -> OptimizationResult:
    optimized = optimize_decision_program(case.program)
    before_terms = _program_ir_terms(case.program)
    after_terms = _program_ir_terms(optimized)
    reduced_terms = before_terms - after_terms
    return OptimizationResult(
        name=case.name,
        scope=case.scope,
        before_terms=before_terms,
        after_terms=after_terms,
        reduced_terms=reduced_terms,
        reduction_rate=reduced_terms / before_terms,
        reason=case.reason,
    )


def _program_ir_terms(program: DecisionProgram) -> int:
    total = 0
    if program.objective is not None:
        total += _expression_ir_terms(program.objective)
    total += sum(
        _expression_ir_terms(constraint.expression)
        for constraint in program.constraints
    )
    return total


def _expression_ir_terms(expression: Expression) -> int:
    constant_terms = 1 if expression.constant != 0.0 else 0
    return len(expression.linear) + len(expression.quadratic) + constant_terms


def _write_acceptance_documents(results: list[OptimizationResult]) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    _write_chart(results)
    (DOC_DIR / "results.md").write_text(_results_document(results), encoding="utf-8")
    (DOC_DIR / "test_report.md").write_text(
        _test_report_document(results), encoding="utf-8"
    )
    (DOC_DIR / "technical_report.md").write_text(
        _technical_report_document(results), encoding="utf-8"
    )


def _results_document(results: list[OptimizationResult]) -> str:
    return "\n".join(
        [
            "# 81 编程框架具有代码自优化功能 - 结果文档",
            "",
            "## 程序输出",
            "",
            "```text",
            _program_output(results),
            "```",
            "",
            "## 结果表格",
            "",
            _summary_table(results),
            "",
            "## 图像导出",
            "",
            "![代码自优化冗余减少图](images/ir_redundancy_reduction.png)",
            "",
        ]
    )


def _test_report_document(results: list[OptimizationResult]) -> str:
    return "\n".join(
        [
            "# 81 编程框架具有代码自优化功能 - 测试报告",
            "",
            "## 测试结论",
            "",
            f"测试通过。4 个用例均出现中间表示冗余减少，总计从 {_total_before(results)} "
            f"个中间表示项降至 {_total_after(results)} 个中间表示项，减少 "
            f"{_total_reduced(results)} 个，整体减少率 {_total_rate(results):.1%}。",
            "",
            "## 测试命令",
            "",
            f"`{TEST_COMMAND}`",
            "",
            "## 覆盖场景分析",
            "",
            "- “目标函数零项消除”覆盖目标函数中零线性项、零二次项和近零常数项的删除。",
            "- “约束零二次项消除”覆盖约束中零二次项的删除，并验证优化后约束可回到线性中间表示。",
            "- “空目标函数压缩”覆盖完全冗余目标函数，验证优化过程可将其压缩为空表达式。",
            "- “混合程序冗余消除”覆盖目标函数和多个约束同时存在冗余项的综合场景。",
            "",
            "## 风险与限制",
            "",
            "当前优化过程只删除数值为零或近零的中间表示项，不做代数重排、公共子表达式消除或语义级重写，"
            "因此不会改变目标函数、约束方向、变量定义和原始 `DecisionProgram` 对象。",
            "",
        ]
    )


def _technical_report_document(results: list[OptimizationResult]) -> str:
    return "\n".join(
        [
            "# 81 编程框架具有代码自优化功能 - 技术报告",
            "",
            "## 技术实现",
            "",
            "代码自优化功能实现为独立编译优化过程，位于 "
            "`src/double_quant/programming/optimizer.py`。该模块不侵入 "
            "`Expression`、`ConstraintExpression` 或 `DecisionProgram` 的核心建模逻辑，"
            "调用方需要显式执行 `optimize_decision_program(program)`。",
            "",
            "## 关键接口",
            "",
            "- `optimize_expression(expression, atol=1.0e-12)`：对单个表达式进行规范化。",
            "- `optimize_constraint(constraint, atol=1.0e-12)`：对约束表达式进行规范化，保留约束方向。",
            "- `optimize_decision_program(program, inplace=False, atol=1.0e-12)`：对决策程序目标函数和约束集合执行统一优化。",
            "",
            "## 中间表示项统计口径",
            "",
            "中间表示项数量定义为：`len(expression.linear) + len(expression.quadratic) + "
            "非零 constant 项数`。该口径直接对应编程框架内部表达式中间表示的结构规模，"
            "不使用源码行数作为评价指标。",
            "",
            "## 优化规则",
            "",
            "优化过程删除绝对值不大于 `1.0e-12` 的线性项、二次项和常数项。"
            "默认 `inplace=False`，因此会复制 `DecisionProgram` 的容器字段并返回优化副本，"
            "避免测试或调用方的原始建模对象被隐式修改。",
            "",
            "## 测试数据摘要",
            "",
            _summary_table(results),
            "",
        ]
    )


def _summary_table(results: list[OptimizationResult]) -> str:
    rows = [
        "| 用例 | 范围 | 优化前中间表示项 | 优化后中间表示项 | 减少冗余 | 减少率 | 减少原因 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        rows.append(
            "| "
            f"{CASE_LABELS[result.name]} | {SCOPE_LABELS[result.scope]} | {result.before_terms} | "
            f"{result.after_terms} | {result.reduced_terms} | "
            f"{result.reduction_rate:.1%} | {result.reason} |"
        )
    rows.append(
        "| "
        f"总计 | {SCOPE_LABELS['all']} | {_total_before(results)} | {_total_after(results)} | "
        f"{_total_reduced(results)} | {_total_rate(results):.1%} | 多用例汇总 |"
    )
    return "\n".join(rows)


def _program_output(results: list[OptimizationResult]) -> str:
    console = Console(
        file=io.StringIO(),
        width=150,
        record=True,
        force_terminal=True,
        color_system=None,
    )
    console.print("功能编号：81")
    console.print("功能名称：编程框架具有代码自优化功能")
    console.print(f"测试命令：{TEST_COMMAND}")
    console.print(
        "汇总结果："
        f"优化前 {_total_before(results)} 个中间表示项，"
        f"优化后 {_total_after(results)} 个中间表示项，"
        f"减少 {_total_reduced(results)} 个冗余项，"
        f"减少率 {_total_rate(results):.1%}"
    )
    console.print()

    table = Table(
        title="代码自优化运行结果",
        box=box.ROUNDED,
        show_lines=True,
        expand=True,
        padding=(0, 1),
    )
    table.add_column("用例", ratio=18, overflow="fold")
    table.add_column("范围", ratio=8, overflow="fold")
    table.add_column("优化前中间表示项", ratio=12, justify="right")
    table.add_column("优化后中间表示项", ratio=12, justify="right")
    table.add_column("减少冗余", ratio=10, justify="right")
    table.add_column("减少率", ratio=10, justify="right")
    table.add_column("减少原因", ratio=34, overflow="fold")
    for result in results:
        table.add_row(
            CASE_LABELS[result.name],
            SCOPE_LABELS[result.scope],
            str(result.before_terms),
            str(result.after_terms),
            str(result.reduced_terms),
            f"{result.reduction_rate:.1%}",
            result.reason,
        )
    table.add_section()
    table.add_row(
        "总计",
        SCOPE_LABELS["all"],
        str(_total_before(results)),
        str(_total_after(results)),
        str(_total_reduced(results)),
        f"{_total_rate(results):.1%}",
        "多用例汇总",
    )
    console.print(table)
    return console.export_text(styles=False)


def _write_chart(results: list[OptimizationResult]) -> None:
    _configure_chinese_font()
    labels = [CASE_LABELS[result.name] for result in results]
    x_positions = list(range(len(results)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5), dpi=160)
    ax.bar(
        [position - width / 2 for position in x_positions],
        [result.before_terms for result in results],
        width=width,
        label="优化前",
        color="#3B82F6",
    )
    ax.bar(
        [position + width / 2 for position in x_positions],
        [result.after_terms for result in results],
        width=width,
        label="优化后",
        color="#10B981",
    )
    ax.set_title("编程框架具有代码自优化功能：中间表示冗余减少")
    ax.set_ylabel("中间表示项数量")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(CHART_PATH)
    plt.close(fig)


def _configure_chinese_font() -> None:
    preferred = [
        "Microsoft YaHei",
        "PingFang SC",
        "Hiragino Sans GB",
        "Hiragino Sans",
        "Heiti SC",
        "STHeiti",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def _total_before(results: list[OptimizationResult]) -> int:
    return sum(result.before_terms for result in results)


def _total_after(results: list[OptimizationResult]) -> int:
    return sum(result.after_terms for result in results)


def _total_reduced(results: list[OptimizationResult]) -> int:
    return sum(result.reduced_terms for result in results)


def _total_rate(results: list[OptimizationResult]) -> float:
    return _total_reduced(results) / _total_before(results)
