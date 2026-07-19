# Func-6 反欺诈监测算法——算法功能测试

## 测试内容

本测试检查 反欺诈监测算法 的算法功能交付是否完整，包括技术报告、应用封装、量子算法电路组件、baseline 组件和关联实验目录。

## 测试结论

- 测试算法：反欺诈监测算法
- 我们的方法：Rasengan 约束环路搜索电路
- baseline：Penalty-QAOA 电路
- 是否测试通过：通过

## 检查项

| 检查项 | 路径 | 是否通过 |
|---|---|---|
| 算法技术报告存在 | third/6-Func-6/out/reports/report.md | 通过 |
| 应用算法封装存在 | src/double_quant/application/antifraud_monitoring.py | 通过 |
| 量子算法电路组件存在 | src/double_quant/algorithm/rasengan/circuit.py | 通过 |
| 量子 baseline 组件存在 | src/double_quant/algorithm/rasengan/baseline.py | 通过 |
| 关联测试目录 16-Func-16 存在 | third/16-Func-16 | 通过 |
| 关联测试目录 26-Func-26 存在 | third/26-Func-26 | 通过 |
| 关联测试目录 36-Func-36 存在 | third/36-Func-36 | 通过 |
| 关联测试目录 47-Perf-6 存在 | third/47-Perf-6 | 通过 |
| 关联测试目录 59-Perf-18 存在 | third/59-Perf-18 | 通过 |
| 关联测试目录 69-Perf-28 存在 | third/69-Perf-28 | 通过 |
| 关联测试目录 125-Perf-46 存在 | third/125-Perf-46 | 通过 |
| 关联测试目录 135-Perf-56 存在 | third/135-Perf-56 | 通过 |

## 结果文件

- ../data/function_test.json
