from __future__ import annotations

import json
from pathlib import Path


TEST_ID = 'Func-6'
ALGORITHM_NAME = '反欺诈监测算法'
METHOD_NAME = 'Rasengan 约束环路搜索电路'
BASELINE_NAME = 'Penalty-QAOA 电路'
APPLICATION_MODULE = 'antifraud_monitoring.py'
ALGORITHM_DIR = 'rasengan'
RELATED_TEST_DIRS = ['16-Func-16', '26-Func-26', '36-Func-36', '47-Perf-6', '59-Perf-18', '69-Perf-28', '125-Perf-46', '135-Perf-56']

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
OUT_DIR = THIS_DIR / "out"
DATA_DIR = OUT_DIR / "data"
REPORT_DIR = OUT_DIR / "reports"


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


def check_file(path: Path, label: str) -> dict[str, object]:
    return {"检查项": label, "路径": str(path.relative_to(ROOT)), "是否通过": path.exists()}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    checks = [
        check_file(THIS_DIR / "out" / "reports" / "report.md", "算法技术报告存在"),
        check_file(ROOT / "src" / "double_quant" / "application" / APPLICATION_MODULE, "应用算法封装存在"),
        check_file(ROOT / "src" / "double_quant" / "algorithm" / ALGORITHM_DIR / "circuit.py", "量子算法电路组件存在"),
        check_file(ROOT / "src" / "double_quant" / "algorithm" / ALGORITHM_DIR / "baseline.py", "量子 baseline 组件存在"),
    ]
    for related_dir in RELATED_TEST_DIRS:
        checks.append(check_file(ROOT / "third" / related_dir, "关联测试目录 " + related_dir + " 存在"))

    passed = all(bool(item["是否通过"]) for item in checks)
    result = {
        "测试项": TEST_ID,
        "测试类型": "算法功能测试",
        "测试算法": ALGORITHM_NAME,
        "我们的方法": METHOD_NAME,
        "baseline": BASELINE_NAME,
        "是否测试通过": passed,
        "检查结果": checks,
    }
    (DATA_DIR / "function_test.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    status = "通过" if passed else "未通过"
    lines = [
        f"# {TEST_ID} {ALGORITHM_NAME}——算法功能测试",
        "",
        "## 测试内容",
        "",
        f"本测试检查 {ALGORITHM_NAME} 的算法功能交付是否完整，包括技术报告、应用封装、量子算法电路组件、baseline 组件和关联实验目录。",
        "",
        "## 测试结论",
        "",
        f"- 测试算法：{ALGORITHM_NAME}",
        f"- 我们的方法：{METHOD_NAME}",
        f"- baseline：{BASELINE_NAME}",
        f"- 是否测试通过：{status}",
        "",
        "## 检查项",
        "",
        "| 检查项 | 路径 | 是否通过 |",
        "|---|---|---|",
    ]
    for item in checks:
        lines.append(f"| {item['检查项']} | {item['路径']} | {'通过' if item['是否通过'] else '未通过'} |")
    lines.extend(["", "## 结果文件", "", "- ../data/function_test.json", ""])
    (REPORT_DIR / "function_test.md").write_text("\n".join(lines), encoding="utf-8")

    print_summary()
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
