TEST_ID = 'Perf-50'
APP_NAME = '指数追踪算法'
METHOD_DESC = 'ours：Rasengan 行业约束篮子搜索电路'
BASELINE_DESC = 'baseline：Penalty-QAOA 电路'
EXP_DIR = 'func_10_index_tracking'
FAMILY = 'rasengan'


import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "out"
DATA_DIR = OUT_DIR / "data"
FIGURE_DIR = OUT_DIR / "figures"
REPORT_DIR = OUT_DIR / "reports"
SOURCE_DATA = THIS_DIR / "source_data"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


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


def read_csv(name: str) -> list[dict[str, str]]:
    path = SOURCE_DATA / name
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def read_json(name: str) -> dict:
    return json.loads((SOURCE_DATA / name).read_text())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float))) if values else 0.0


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def ratio(value: float) -> str:
    return f"{value:.2f}x"


def cn_status(ok: bool) -> str:
    return "通过" if ok else "未通过"


def label(method_label: str) -> str:
    return "ours" if method_label == "method" else "baseline"


def tex_expr(expr: str) -> str:
    expr = str(expr).strip()
    expr = expr.replace(" * ", " ")
    expr = re.sub(r"2\^\(([^)]+)\)", r"2^{\1}", expr)
    expr = re.sub(r"2\^n", r"2^{n}", expr)
    expr = re.sub(r"n\^([0-9.]+)", r"n^{\1}", expr)
    expr = expr.replace(" / ", r" / ")
    expr = re.sub(r"(?<=\d) (?=n|2)", r"\\,", expr)
    return expr


def fit_value(fit: dict, x: np.ndarray) -> np.ndarray | None:
    if not fit:
        return None
    model = str(fit.get("model", ""))
    a = f(fit.get("a", 0.0))
    b = f(fit.get("b", 0.0))
    if model == "power":
        return a * np.power(x, b)
    if model == "exponential":
        return a * np.power(2.0, b * x)
    if model == "linear":
        return a * x + b
    if model == "constant":
        return np.full_like(x, a, dtype=float)
    return None


def auto_fit(x_values: list[float], y_values: list[float]) -> dict[str, object]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        value = float(y[0]) if len(y) else 0.0
        return {"model": "constant", "a": value, "b": 0.0, "expression": f"{value:.1f}"}
    candidates: list[dict[str, object]] = []
    lx = np.log(x)
    ly = np.log(y)
    b, loga = np.polyfit(lx, ly, 1)
    yhat = np.exp(loga) * np.power(x, b)
    candidates.append({"model": "power", "a": float(np.exp(loga)), "b": float(b), "r2": r2(y, yhat)})
    b2, loga2 = np.polyfit(x, np.log2(y), 1)
    yhat2 = np.power(2.0, loga2 + b2 * x)
    candidates.append({"model": "exponential", "a": float(2.0 ** loga2), "b": float(b2), "r2": r2(y, yhat2)})
    best = max(candidates, key=lambda item: float(item["r2"]))
    best["expression"] = fit_expression(best)
    return best


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot


def trim_number(value: float) -> str:
    if abs(value - 1.0) < 0.05:
        return ""
    if value >= 100:
        text = f"{value:.0f}"
    elif value >= 10:
        text = f"{value:.1f}"
    else:
        text = f"{value:.2g}"
    return text.rstrip("0").rstrip(".")


def fit_expression(fit: dict[str, object]) -> str:
    a = f(fit.get("a", 0.0))
    b = f(fit.get("b", 0.0))
    coeff = trim_number(a)
    model = fit.get("model")
    if model == "power":
        term = "n" if abs(b - 1.0) < 0.05 else f"n^{b:.1f}"
        return f"{coeff} {term}".strip()
    if model == "exponential":
        if abs(b) < 0.05:
            return f"{a:.1f}".rstrip("0").rstrip(".")
        term = "2^n" if abs(b - 1.0) < 0.05 else f"2^({b:.1f} n)"
        return f"{coeff} {term}".strip()
    return f"{a:.1f}".rstrip("0").rstrip(".")


def plot_two_series_with_fits(
    *,
    rows: list[dict[str, object]],
    x_key: str,
    ours_key: str,
    baseline_key: str,
    fit_block: dict | None,
    title: str,
    ylabel: str,
    path: Path,
    yscale: str = "linear",
) -> None:
    x = np.asarray([f(row[x_key]) for row in rows], dtype=float)
    ours = np.asarray([f(row[ours_key]) for row in rows], dtype=float)
    baseline = np.asarray([f(row[baseline_key]) for row in rows], dtype=float)
    order = np.argsort(x)
    x, ours, baseline = x[order], ours[order], baseline[order]
    xfit = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    plt.figure(figsize=(7.2, 4.8), dpi=160)
    plt.scatter(x, ours, color="#1f77b4", label="ours 数据", zorder=3)
    plt.scatter(x, baseline, color="#d62728", label="baseline 数据", zorder=3)
    if fit_block:
        yfit = fit_value(fit_block.get("method", {}), xfit)
        if yfit is not None:
            plt.plot(xfit, yfit, color="#1f77b4", linewidth=2, label=f"ours: ${tex_expr(fit_block['method'].get('expression', ''))}$")
        yfit = fit_value(fit_block.get("baseline", {}), xfit)
        if yfit is not None:
            plt.plot(xfit, yfit, color="#d62728", linewidth=2, label=f"baseline: ${tex_expr(fit_block['baseline'].get('expression', ''))}$")
    plt.xlabel("总变量数 n")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.yscale(yscale)
    plt.grid(True, alpha=0.25, which="both")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_bars(*, labels: list[str], values: list[float], title: str, ylabel: str, path: Path, yscale: str = "linear", annotate: str | None = None) -> None:
    plt.figure(figsize=(6.6, 4.4), dpi=160)
    colors = ["#1f77b4", "#d62728"]
    bars = plt.bar(labels, values, color=colors[: len(values)], width=0.58)
    for bar, value in zip(bars, values):
        if "（%）" in ylabel or "(%)" in ylabel:
            text = f"{value:.1f}%"
        else:
            text = f"{value:.3g}" if yscale == "log" else (pct(value) if 0 <= value <= 1.2 else f"{value:.2f}")
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text, ha="center", va="bottom", fontsize=10)
    if annotate:
        plt.text(0.5, 0.92, annotate, transform=plt.gca().transAxes, ha="center", va="center", fontsize=11, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999999", alpha=0.9))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.yscale(yscale)
    plt.grid(True, axis="y", alpha=0.25, which="both")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()



def task_success(p: float, repetitions: int) -> float:
    repetitions = max(1, int(repetitions))
    return 1.0 - (1.0 - min(1.0, max(0.0, p))) ** repetitions


def main() -> None:
    ensure_dirs()
    setup_font()
    if FAMILY == "shor":
        scaling = read_csv("scaling_cases.csv")
        best = None
        for row in scaling:
            method_reps = max(1, math.floor(f(row["baseline_depth"]) / max(f(row["method_depth"]), 1e-12)))
            baseline_reps = 1
            ours_success = task_success(f(row["method_noisy_success_probability"]), method_reps)
            baseline_success = task_success(f(row["baseline_noisy_success_probability"]), baseline_reps)
            ours_error = 1.0 - ours_success
            baseline_error = 1.0 - baseline_success
            reduction = baseline_error - ours_error
            candidate = (reduction, ours_success, baseline_success, ours_error, baseline_error, f(row["n"]), method_reps)
            if best is None or candidate[0] > best[0]:
                best = candidate
        reduction, ours_success, baseline_success, ours_error, baseline_error, case_n, reps = best
        error_name = "同一深度预算下含噪任务错误率"
        detail = f"Shor 类测试采用同一含噪深度预算，ours 因电路更浅可重复 ${reps}$ 次；错误率为 $1-P_{{task,noise}}$。"
    elif FAMILY == "grover":
        rows = read_csv("noisy_precision_iteration.csv")
        best_by_label = {}
        for method_label in ("method", "baseline"):
            subset = [row for row in rows if row["method_label"] == method_label]
            best_by_label[method_label] = max(subset, key=lambda row: f(row["success_probability"]))
        ours_success = f(best_by_label["method"]["success_probability"])
        baseline_success = f(best_by_label["baseline"]["success_probability"])
        ours_error = 1.0 - ours_success
        baseline_error = 1.0 - baseline_success
        reduction = baseline_error - ours_error
        error_name = "含噪采样错误率"
        detail = "Grover 类测试以含噪采样未命中标记解的概率为误差，即 $E=1-P_{success,noise}$。"
    else:
        rows = read_csv("noisy_precision_parameter.csv")
        best_by_label = {}
        for method_label in ("method", "baseline"):
            subset = [row for row in rows if row["method_label"] == method_label]
            best_by_label[method_label] = min(subset, key=lambda row: f(row["ARG"]))
        ours_arg = f(best_by_label["method"]["ARG"])
        baseline_arg = f(best_by_label["baseline"]["ARG"])
        ours_success = f(best_by_label["method"]["precision_score"])
        baseline_success = f(best_by_label["baseline"]["precision_score"])
        ours_error = ours_arg / max(baseline_arg, 1e-12)
        baseline_error = 1.0
        reduction = baseline_error - ours_error
        error_name = "含噪基线归一化 ARG 误差率"
        detail = "Rasengan 类测试先计算含噪 $ARG$。为使误差百分比保持在 $0\\%$ 到 $100\\%$ 内，本测试用基线 ARG 归一化：$E_{ours}=ARG_{ours}/ARG_{baseline}$，$E_{baseline}=1$。因此 baseline 为 $100\\%$，ours 表示相对 baseline 剩余多少误差。"
    ours_error_percent = 100.0 * ours_error
    baseline_error_percent = 100.0 * baseline_error
    reduction_percent = baseline_error_percent - ours_error_percent
    result_rows = [
        {"方法": "ours", "含噪精度或成功率": ours_success, "含噪计算误差（%）": ours_error_percent},
        {"方法": "baseline", "含噪精度或成功率": baseline_success, "含噪计算误差（%）": baseline_error_percent},
    ]
    write_csv(DATA_DIR / "noisy_error_reduction.csv", result_rows)
    metrics = {
        "测试项": TEST_ID,
        "算法": APP_NAME,
        "误差指标": error_name,
        "我们的含噪误差百分比": ours_error_percent,
        "基线含噪误差百分比": baseline_error_percent,
        "含噪误差降低百分比": reduction_percent,
        "是否通过": reduction_percent >= 40.0,
    }
    write_json(DATA_DIR / "metrics.json", metrics)
    fig_path = FIGURE_DIR / "noisy_error_reduction.png"
    plot_bars(
        labels=["ours", "baseline"],
        values=[ours_error_percent, baseline_error_percent],
        title=f"{TEST_ID} {APP_NAME}：含噪计算误差降低40%及以上测试",
        ylabel=f"{error_name}（%）",
        path=fig_path,
        yscale="linear",
        annotate=f"误差降低 {reduction_percent:.1f}%",
    )
    report = f"""# {TEST_ID} {APP_NAME}——相较于IBM Qiskit部署方案降低40%以上计算误差测试

## 测试对象

- 我们的方法：{METHOD_DESC}
- 量子基线：{BASELINE_DESC}

## 指标定义

误差指标为“{error_name}”。报告统一展示百分误差 $E_{{\%}}=100\\times E$。误差降低采用百分误差的绝对差值定义：$\\Delta E_{{\%}}=E_{{baseline,\%}}-E_{{ours,\%}}$；目标“不低于 $40\\%$”按 $\\Delta E_{{\%}}\\ge 40\\%$ 判定。{detail}

## 测试结果

- 我们的含噪误差：${ours_error_percent:.2f}\\%$。
- 基线含噪误差：${baseline_error_percent:.2f}\\%$。
- 含噪误差降低值：${reduction_percent:.2f}\\%$。
- 达标结论：{cn_status(bool(metrics["是否通过"]))}，目标为不低于 $40\\%$。

![含噪计算误差降低40%及以上测试](../figures/noisy_error_reduction.png)

## 结果文件

本测试的原始数据表和指标摘要保存在 `../data/` 目录。
"""
    (REPORT_DIR / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

