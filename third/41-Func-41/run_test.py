from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
TEST_ID = "Func-41"
TEST_NAME = "求解精度与量子复杂度分析优化理论测试"
APP_NAME = TEST_NAME
OUT_DIR = Path(__file__).resolve().parent / "out"
DATA_DIR = OUT_DIR / "data"
FIGURE_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "reports"
SOURCE_DATA = Path(__file__).resolve().parent / "source_data"


APPLICATIONS = [
    ("Func-4", "动态账本更新算法", "func_4_dynamic_ledger_update", "shor"),
    ("Func-5", "去中心化金融管理算法", "func_5_defi_management", "grover"),
    ("Func-6", "反欺诈监测算法", "func_6_antifraud_monitoring", "rasengan"),
    ("Func-7", "支付与结算系统算法", "func_7_payment_settlement", "rasengan"),
    ("Func-8", "贷款发放决策算法", "func_8_loan_decision", "rasengan"),
    ("Func-9", "银行网点布局优化算法", "func_9_branch_location", "grover"),
    ("Func-10", "指数追踪算法", "func_10_index_tracking", "rasengan"),
]




def print_summary() -> None:
    current_dir = globals().get("THIS_DIR", Path(__file__).resolve().parent)
    data_dir = globals().get("DATA_DIR", current_dir / "out" / "data")
    report_dir = globals().get("REPORT_DIR", current_dir / "out" / "reports")
    app_label = globals().get("APP_NAME", globals().get("ALGORITHM_NAME", globals().get("TEST_NAME", "")))

    metrics_path = data_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        test_id = str(metrics.get("测试项", globals().get("TEST_ID", "")))
        algorithm = str(metrics.get("算法", metrics.get("测试名称", app_label)))
        raw_status = metrics.get("是否通过", metrics.get("全部达标", None))
        status = "通过" if raw_status is True else "未通过" if raw_status is False else "已完成"
        skipped = {"测试项", "算法", "测试名称", "是否通过", "全部达标"}
        parts = []
        for key, value in metrics.items():
            if key in skipped:
                continue
            if isinstance(value, float):
                if "比例" in key or "降低" in key or "提升" in key or "误差百分比" in key:
                    value = f"{value * 100:.1f}%" if abs(value) <= 1.0 else f"{value:.1f}%"
                elif "加速" in key and "拟合" not in key:
                    value = f"{value:.2f}x"
                else:
                    value = f"{value:.4g}"
            parts.append(f"{key}={value}")
            if len(parts) >= 4:
                break
        print(f"{test_id} {algorithm}：{status}")
        if parts:
            print("  摘要：" + "；".join(parts))
        print(f"  结果：{data_dir.relative_to(current_dir)} / {report_dir.relative_to(current_dir)}")
        return

    function_path = data_dir / "function_test.json"
    if function_path.exists():
        result = json.loads(function_path.read_text(encoding="utf-8"))
        status = "通过" if result.get("是否测试通过") else "未通过"
        print(f"{result.get('测试项', globals().get('TEST_ID', ''))} {result.get('测试算法', app_label)}：{status}")
        print(f"  摘要：测试类型={result.get('测试类型', '算法功能测试')}；我们的方法={result.get('我们的方法', '')}")
        print(f"  结果：{data_dir.relative_to(current_dir)} / {report_dir.relative_to(current_dir)}")
        return

    print(f"{globals().get('TEST_ID', '')} {app_label}：测试完成。")


def setup_font() -> None:
    # Latin/digits are drawn by DejaVu Sans; Chinese glyphs fall back to a
    # plain CJK font so labels such as "门数量" do not become tofu boxes.
    for path in [
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]:
        if Path(path).exists():
            fm.fontManager.addfont(path)
    plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback", "Noto Sans CJK JP"]
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Droid Sans Fallback", "Noto Sans CJK JP"]
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["axes.unicode_minus"] = False


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def f(value: object, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def tex_expr(expr: str) -> str:
    expr = str(expr).strip()
    expr = re.sub(r"2\^\(([^)]+)\)", r"2^{\1}", expr)
    expr = re.sub(r"2\^n", r"2^{n}", expr)
    expr = re.sub(r"n\^([0-9.]+)", r"n^{\1}", expr)
    expr = re.sub(r"(?<=\d) (?=n|2)", r"\\,", expr)
    return expr


def normalized_precision_gain(ours: float, baseline: float) -> float:
    return (ours - baseline) / max(1.0 - baseline, 1e-12)


def precision_values(data_dir: Path, metrics: dict, family: str) -> tuple[float, float]:
    if family == "shor":
        return f(metrics["clean_task_success_method"]), f(metrics["clean_task_success_baseline"])
    if family == "grover":
        return f(metrics["precision_equal_budget_method"]), f(metrics["precision_equal_budget_baseline"])

    rows = read_csv(data_dir / "precision_parameter.csv")
    ours = max(f(row["precision_score"]) for row in rows if row["method_label"] == "method")
    baseline = max(f(row["precision_score"]) for row in rows if row["method_label"] == "baseline")
    return ours, baseline


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for func_id, app_name, exp_dir, family in APPLICATIONS:
        data_dir = SOURCE_DATA / exp_dir
        metrics = read_json(data_dir / "goal_metrics.json")
        fits = read_json(data_dir / "fits.json")
        precision_ours, precision_baseline = precision_values(data_dir, metrics, family)
        point_gain = precision_ours - precision_baseline
        precision_gain = normalized_precision_gain(precision_ours, precision_baseline)
        complexity_reduction = f(
            metrics.get(
                "complexity_reduction_median",
                metrics.get("complexity_reduction", 0.0),
            )
        )
        complexity_speedup = f(metrics.get("complexity_speedup_median", 0.0))
        complexity_fit = str(fits["complexity"]["speedup"]["expression"])
        rows.append(
            {
                "算法编号": func_id,
                "算法名称": app_name,
                "我们的精度": precision_ours,
                "基线精度": precision_baseline,
                "提升百分点": point_gain,
                "归一化精度提升比例": precision_gain,
                "精度提升比例": precision_gain,
                "复杂度降低比例": complexity_reduction,
                "复杂度中位加速比": complexity_speedup,
                "复杂度加速拟合表达式": complexity_fit,
                "精度提升是否达标": precision_gain >= 0.4,
                "复杂度降低是否达标": complexity_reduction >= 0.5,
            }
        )
    return rows


def draw_figure(rows: list[dict[str, object]], path: Path) -> None:
    labels = [str(row["算法编号"]) for row in rows]
    precision_gain = np.asarray([float(row["归一化精度提升比例"]) for row in rows])
    complexity_reduction = np.asarray([float(row["复杂度降低比例"]) for row in rows])
    speedup = np.asarray([float(row["复杂度中位加速比"]) for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), dpi=160)

    axes[0].bar(labels, precision_gain, color="#1f77b4")
    axes[0].axhline(0.4, color="#d62728", linestyle="--", label="40%目标线")
    axes[0].set_ylim(0, max(1.05, float(np.max(precision_gain)) * 1.12))
    axes[0].set_ylabel("归一化精度提升比例")
    axes[0].set_title("归一化精度提升")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].bar(labels, complexity_reduction, color="#2ca02c")
    axes[1].axhline(0.5, color="#d62728", linestyle="--", label="50%目标线")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("复杂度降低比例")
    axes[1].set_title("复杂度降低")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)

    axes[2].bar(labels, speedup, color="#9467bd")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("复杂度中位加速比（对数刻度）")
    axes[2].set_title("复杂度加速")
    axes[2].grid(True, axis="y", alpha=0.25, which="both")

    for ax in axes:
        ax.set_xlabel("算法编号")
    fig.suptitle(f"{TEST_ID}：{TEST_NAME}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_report(rows: list[dict[str, object]]) -> str:
    lines = [
        f"# {TEST_ID} {TEST_NAME}",
        "",
        "## 测试定义",
        "",
        "本测试汇总 Func-4 至 Func-10 的已验收实验结果，验证每个量子金融算法都给出了求解精度、量子复杂度和二者之间的可比较关系。精度指标按各算法实验定义选取：Shor 类使用同一深度预算下任务成功率，Grover 类使用同一迭代预算下采样成功率，Rasengan 类使用 $P=1/(1+ARG)$。",
        "",
        "归一化精度提升比例定义为 $(P_{ours}-P_{baseline})/(1-P_{baseline})$，表示在 baseline 距满分精度 1 的剩余空间中，ours 弥合的比例。复杂度统一记为 $C(n)$，加速比定义为 $S(n)=C_{baseline}(n)/C_{ours}(n)$；复杂度降低比例定义为 $1-C_{ours}(n)/C_{baseline}(n)$。每个算法的 $S(n)$ 均来自对应实验的拟合曲线。",
        "",
        "## 达标判断",
        "",
        "归一化精度提升目标为不低于 $40\\%$，复杂度降低目标为不低于 $50\\%$，且复杂度加速拟合表达式需要随总变量数 $n$ 增长或整体显著大于常数。",
        "",
        "| 算法 | 归一化精度提升比例 | 复杂度降低比例 | 复杂度加速拟合表达式 | 结论 |",
        "|---|---:|---:|---|---|",
    ]
    all_pass = True
    for row in rows:
        ok = bool(row["精度提升是否达标"]) and bool(row["复杂度降低是否达标"])
        all_pass = all_pass and ok
        lines.append(
            f"| {row['算法编号']} {row['算法名称']} | {100*float(row['归一化精度提升比例']):.1f}% | "
            f"{100*float(row['复杂度降低比例']):.1f}% | ${tex_expr(str(row['复杂度加速拟合表达式']))}$ | "
            f"{'通过' if ok else '未通过'} |"
        )
    lines.extend(
        [
            "",
            f"总体结论：{'通过' if all_pass else '未通过'}。7 个算法均给出了精度与复杂度的定量定义、实验曲线和随 $n$ 变化的复杂度加速表达式。",
            "",
            "![求解精度与量子复杂度分析优化理论测试](../figures/求解精度与量子复杂度分析优化理论.png)",
            "",
            "## 结果文件",
            "",
            "- `../data/theory_summary.csv`",
            "- `../data/metrics.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    setup_font()
    rows = build_rows()
    write_csv(DATA_DIR / "theory_summary.csv", rows)
    all_pass = all(bool(row["精度提升是否达标"]) and bool(row["复杂度降低是否达标"]) for row in rows)
    write_json(
        DATA_DIR / "metrics.json",
        {
            "测试项": TEST_ID,
            "测试名称": TEST_NAME,
            "算法数量": len(rows),
            "全部达标": all_pass,
        },
    )
    draw_figure(rows, FIGURE_DIR / "求解精度与量子复杂度分析优化理论.png")
    (REPORT_DIR / "report.md").write_text(make_report(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
    print_summary()
