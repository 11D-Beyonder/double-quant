# Func-9 银行网点布局优化算法——算法功能测试

## 测试内容

本测试检查 银行网点布局优化算法 的算法功能交付是否完整，包括技术报告、应用封装、量子算法电路组件、baseline 组件和关联实验目录。

## 测试结论

- 测试算法：银行网点布局优化算法
- 我们的方法：SFS-Grover 设施选址搜索电路
- baseline：普通 Grover 量子搜索电路
- 是否测试通过：通过

## 检查项

| 检查项 | 路径 | 是否通过 |
|---|---|---|
| 算法技术报告存在 | third/9-Func-9/out/reports/report.md | 通过 |
| 应用算法封装存在 | src/double_quant/application/branch_location.py | 通过 |
| 量子算法电路组件存在 | src/double_quant/algorithm/grover/circuit.py | 通过 |
| 量子 baseline 组件存在 | src/double_quant/algorithm/grover/baseline.py | 通过 |
| 关联测试目录 19-Func-19 存在 | third/19-Func-19 | 通过 |
| 关联测试目录 29-Func-29 存在 | third/29-Func-29 | 通过 |
| 关联测试目录 39-Func-39 存在 | third/39-Func-39 | 通过 |
| 关联测试目录 50-Perf-9 存在 | third/50-Perf-9 | 通过 |
| 关联测试目录 62-Perf-21 存在 | third/62-Perf-21 | 通过 |
| 关联测试目录 72-Perf-31 存在 | third/72-Perf-31 | 通过 |
| 关联测试目录 128-Perf-49 存在 | third/128-Perf-49 | 通过 |
| 关联测试目录 138-Perf-59 存在 | third/138-Perf-59 | 通过 |

## 结果文件

- ../data/function_test.json
