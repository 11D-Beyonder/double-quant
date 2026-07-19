# Func-4 动态账本更新算法——算法功能测试

## 测试内容

本测试检查 动态账本更新算法 的算法功能交付是否完整，包括技术报告、应用封装、量子算法电路组件、baseline 组件和关联实验目录。

## 测试结论

- 测试算法：动态账本更新算法
- 我们的方法：优化动态账本 Shor 周期发现电路
- baseline：通用 Shor 周期发现量子基线
- 是否测试通过：通过

## 检查项

| 检查项 | 路径 | 是否通过 |
|---|---|---|
| 算法技术报告存在 | third/4-Func-4/out/reports/report.md | 通过 |
| 应用算法封装存在 | src/double_quant/application/dynamic_ledger_update.py | 通过 |
| 量子算法电路组件存在 | src/double_quant/algorithm/shor/circuit.py | 通过 |
| 量子 baseline 组件存在 | src/double_quant/algorithm/shor/baseline.py | 通过 |
| 关联测试目录 14-Func-14 存在 | third/14-Func-14 | 通过 |
| 关联测试目录 24-Func-24 存在 | third/24-Func-24 | 通过 |
| 关联测试目录 34-Func-34 存在 | third/34-Func-34 | 通过 |
| 关联测试目录 45-Perf-4 存在 | third/45-Perf-4 | 通过 |
| 关联测试目录 57-Perf-16 存在 | third/57-Perf-16 | 通过 |
| 关联测试目录 67-Perf-26 存在 | third/67-Perf-26 | 通过 |
| 关联测试目录 123-Perf-44 存在 | third/123-Perf-44 | 通过 |
| 关联测试目录 133-Perf-54 存在 | third/133-Perf-54 | 通过 |

## 结果文件

- ../data/function_test.json
