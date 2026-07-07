TEST_ID = 'Func-37'
APP_NAME = '支付与结算系统算法'
METHOD_DESC = 'ours：Rasengan 流动性中性搜索电路'
BASELINE_DESC = 'baseline：Penalty-QAOA 电路'
EXP_DIR = 'func_7_payment_settlement'
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
    source_file = "success_parameter.csv" if FAMILY == "shor" else ("precision_iteration.csv" if FAMILY == "grover" else "precision_parameter.csv")
    rows = read_csv(source_file)
    metric_key = "precision_score" if FAMILY == "rasengan" else "success_probability"
    x_label = "相位寄存器量子位数 t" if FAMILY == "shor" else ("Grover 迭代次数 r" if FAMILY == "grover" else "COBYLA 迭代次数")
    y_label = "精度得分 P=1/(1+ARG)" if FAMILY == "rasengan" else "成功率/精度"
    result_rows = []
    by_param: dict[float, dict[str, float]] = {}
    for row in rows:
        p = f(row["parameter_value"])
        by_param.setdefault(p, {"参数值": p})
        by_param[p][label(row["method_label"])] = f(row[metric_key])
    for p in sorted(by_param):
        result_rows.append({
            "量子电路参数": p,
            "我们的精度": by_param[p].get("ours", ""),
            "基线精度": by_param[p].get("baseline", ""),
        })
    write_csv(DATA_DIR / "precision_parameter_relation.csv", result_rows)
    metrics = {
        "测试项": TEST_ID,
        "算法": APP_NAME,
        "参数名称": x_label,
        "精度指标": y_label,
        "数据点数量": len(result_rows),
        "是否通过": len(result_rows) >= 2,
    }
    write_json(DATA_DIR / "metrics.json", metrics)
    fig_path = FIGURE_DIR / "精度与量子电路参数关系.png"
    if FAMILY == "rasengan":
        fig, (upper_axis, lower_axis) = plt.subplots(2, 1, sharex=True, figsize=(7.2, 5.2), dpi=160, gridspec_kw={"height_ratios": [2.5, 1.4], "hspace": 0.06})
        plotted = []
        for method_label, color in [("method", "#1f77b4"), ("baseline", "#d62728")]:
            subset = [row for row in rows if row["method_label"] == method_label]
            x = [f(row["parameter_value"]) for row in subset]
            y = [f(row[metric_key]) for row in subset]
            plotted.extend(y)
            for ax in (upper_axis, lower_axis):
                ax.plot(x, y, marker="o", linewidth=2, color=color, label=label(method_label))
        vals = sorted(v for v in plotted if math.isfinite(v))
        lower_axis.set_ylim(max(0.0, min(vals) * 0.8), max(0.05, np.percentile(vals, 35) * 1.15))
        upper_axis.set_ylim(max(0.0, np.percentile(vals, 55) * 0.95), min(1.03, max(vals) * 1.05 + 0.02))
        upper_axis.spines.bottom.set_visible(False)
        lower_axis.spines.top.set_visible(False)
        upper_axis.tick_params(labeltop=False, bottom=False)
        lower_axis.xaxis.tick_bottom()
        diagonal_kwargs = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=10, linestyle="none", color="k", mec="k", mew=1, clip_on=False)
        upper_axis.plot([0, 1], [0, 0], transform=upper_axis.transAxes, **diagonal_kwargs)
        lower_axis.plot([0, 1], [1, 1], transform=lower_axis.transAxes, **diagonal_kwargs)
        upper_axis.set_title(f"{TEST_ID} {APP_NAME}：精度与量子电路参数关系")
        lower_axis.set_xlabel(x_label)
        fig.supylabel(y_label)
        for ax in (upper_axis, lower_axis):
            ax.grid(True, alpha=0.25)
        upper_axis.legend()
        fig.subplots_adjust(hspace=0.06, left=0.13, right=0.97, top=0.91, bottom=0.11)
        fig.savefig(fig_path)
        plt.close(fig)
    else:
        plt.figure(figsize=(7.2, 4.8), dpi=160)
        for method_label, color in [("method", "#1f77b4"), ("baseline", "#d62728")]:
            subset = [row for row in rows if row["method_label"] == method_label]
            x = [f(row["parameter_value"]) for row in subset]
            y = [f(row[metric_key]) for row in subset]
            plt.plot(x, y, marker="o", linewidth=2, color=color, label=label(method_label))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.ylim(-0.03, 1.03)
        plt.title(f"{TEST_ID} {APP_NAME}：精度与量子电路参数关系")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
    metric_define = "Rasengan/Penalty-QAOA 使用 $P=1/(1+ARG)$，其中 $ARG$ 按概率加权的惩罚目标函数计算，不满足约束的样本计入惩罚。" if FAMILY == "rasengan" else "Shor 与 Grover 类测试使用采样成功率作为精度，成功率由真实 Qiskit 电路采样结果统计得到。"
    report = f"""# {TEST_ID} {APP_NAME}——计算精度与量子电路参数之间的函数关系测试

## 测试对象

- 我们的方法：{METHOD_DESC}
- 量子基线：{BASELINE_DESC}

## 指标定义

量子电路参数为：{x_label}。精度指标为：{y_label}。{metric_define}

## 测试结果

- 曲线数据点数量：${metrics["数据点数量"]}$。
- 达标结论：{cn_status(bool(metrics["是否通过"]))}，已给出精度随量子电路参数变化的定量曲线与数据表。

![计算精度与量子电路参数之间的函数关系测试](../figures/精度与量子电路参数关系.png)

## 结果文件

本测试的原始数据表和指标摘要保存在 `../data/` 目录。
"""
    (REPORT_DIR / "report.md").write_text(report, encoding="utf-8")
    print_summary()


if __name__ == "__main__":
    main()

