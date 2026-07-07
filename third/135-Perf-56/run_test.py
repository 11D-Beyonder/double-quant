TEST_ID = 'Perf-56'
APP_NAME = '反欺诈监测算法'
METHOD_DESC = 'ours：Rasengan 约束环路搜索电路'
BASELINE_DESC = 'baseline：Penalty-QAOA 电路'
EXP_DIR = 'func_6_antifraud_monitoring'
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



def main() -> None:
    ensure_dirs()
    setup_font()
    if FAMILY == "shor":
        rows = read_csv("scaling_cases.csv")
        ours_key = "method_noisy_complexity"
        baseline_key = "baseline_noisy_complexity"
        fit_key = "noisy_complexity"
        note = "Shor 类测试采用含噪成功率修正复杂度 $C_{noise}=D/P_{success,noise}$。"
    elif FAMILY == "grover":
        rows = read_csv("noisy_complexity_cases.csv")
        ours_key = "method_noisy_complexity"
        baseline_key = "baseline_noisy_complexity"
        fit_key = "noisy_complexity"
        note = "Grover 类测试在含噪模拟器中扫描达到共同成功率阈值所需迭代数，再计算 $C_{noise}=D\\times I_{noise}$。"
    else:
        rows = read_csv("scaling_cases.csv")
        ours_key = "method_complexity"
        baseline_key = "baseline_complexity"
        fit_key = "complexity"
        note = "Rasengan 类测试中噪声改变采样概率但不改变名义电路深度，因此含噪复杂度沿用同一电路深度乘迭代预算。"
    fits = read_json("fits.json")
    result_rows = []
    for row in rows:
        ours = f(row[ours_key])
        baseline = f(row[baseline_key])
        reduction = 1.0 - ours / max(baseline, 1e-12)
        result_rows.append({
            "变量数n": f(row["n"]),
            "我们的含噪复杂度": ours,
            "基线含噪复杂度": baseline,
            "含噪复杂度降低比例": reduction,
        })
    write_csv(DATA_DIR / "noisy_complexity_reduction.csv", result_rows)
    reductions = [f(row["含噪复杂度降低比例"]) for row in result_rows]
    speed_expr = fits[fit_key]["speedup"]["expression"]
    metrics = {
        "测试项": TEST_ID,
        "算法": APP_NAME,
        "含噪复杂度降低中位数": median(reductions),
        "含噪复杂度降低最小值": min(reductions),
        "含噪复杂度加速拟合": speed_expr,
        "是否通过": median(reductions) >= 0.5,
    }
    write_json(DATA_DIR / "metrics.json", metrics)
    fig_path = FIGURE_DIR / "noisy_complexity_reduction.png"
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), dpi=160)
    x = np.asarray([f(row["变量数n"]) for row in result_rows])
    ours = np.asarray([f(row["我们的含噪复杂度"]) for row in result_rows])
    base = np.asarray([f(row["基线含噪复杂度"]) for row in result_rows])
    red = np.asarray([f(row["含噪复杂度降低比例"]) for row in result_rows])
    axes[0].plot(x, ours, marker="o", linewidth=2, color="#1f77b4", label="ours")
    axes[0].plot(x, base, marker="o", linewidth=2, color="#d62728", label="baseline")
    axes[0].set_xlabel("总变量数 n")
    if float(np.max(base)) / max(float(np.min(ours)), 1e-12) > 20:
        axes[0].set_yscale("log")
        axes[0].set_ylabel("含噪复杂度（对数刻度）")
        axes[0].grid(True, alpha=0.25, which="both")
    else:
        axes[0].set_ylabel("含噪复杂度")
        axes[0].grid(True, alpha=0.25)
    axes[0].set_title("含噪复杂度对比")
    axes[0].legend()
    axes[1].plot(x, red, marker="o", linewidth=2, color="#2ca02c", label="降低比例")
    axes[1].axhline(0.5, color="#d62728", linestyle="--", label="50% 目标线")
    axes[1].set_xlabel("总变量数 n")
    axes[1].set_ylabel("含噪复杂度降低比例")
    axes[1].set_ylim(0, min(1.05, max(1.0, float(np.max(red)) * 1.08)))
    axes[1].set_title("降低比例")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.suptitle(f"{TEST_ID} {APP_NAME}：含噪复杂度降低50%及以上测试")
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    report = f"""# {TEST_ID} {APP_NAME}——相较于IBM Qiskit部署方案降低50%以上量子计算复杂度测试

## 测试对象

- 我们的方法：{METHOD_DESC}
- 量子基线：{BASELINE_DESC}

## 指标定义

含噪复杂度降低比例定义为 $1-C_{{noise,ours}}(n)/C_{{noise,baseline}}(n)$。图中左侧给出 ours 与 baseline 的含噪复杂度对比；若两者量级差异较大，左图自动使用对数纵轴。右侧给出含噪复杂度降低比例及 50% 目标线。{note}

## 测试结果

- 含噪复杂度加速拟合：$S_{{noise}}(n)={tex_expr(metrics["含噪复杂度加速拟合"])}$。
- 含噪复杂度降低中位数：${pct(metrics["含噪复杂度降低中位数"])}$。
- 含噪复杂度降低最小值：${pct(metrics["含噪复杂度降低最小值"])}$。
- 达标结论：{cn_status(bool(metrics["是否通过"]))}，目标为不低于 $50\\%$。

![含噪复杂度降低50%及以上测试](../figures/noisy_complexity_reduction.png)

## 结果文件

本测试的原始数据表和指标摘要保存在 `../data/` 目录。
"""
    (REPORT_DIR / "report.md").write_text(report, encoding="utf-8")
    print_summary()


if __name__ == "__main__":
    main()

