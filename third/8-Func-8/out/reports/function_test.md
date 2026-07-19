# Func-8 贷款发放决策算法——算法功能测试

## 测试内容

本测试检查 贷款发放决策算法 的算法功能交付是否完整，包括技术报告、应用封装、量子算法电路组件、baseline 组件和关联实验目录。

## 测试结论

- 测试算法：贷款发放决策算法
- 我们的方法：Rasengan 分组特征选择电路
- baseline：Penalty-QAOA 电路
- 是否测试通过：通过

## 检查项

| 检查项 | 路径 | 是否通过 |
|---|---|---|
| 算法技术报告存在 | third/8-Func-8/out/reports/report.md | 通过 |
| 应用算法封装存在 | src/double_quant/application/loan_decision.py | 通过 |
| 量子算法电路组件存在 | src/double_quant/algorithm/rasengan/circuit.py | 通过 |
| 量子 baseline 组件存在 | src/double_quant/algorithm/rasengan/baseline.py | 通过 |
| 关联测试目录 18-Func-18 存在 | third/18-Func-18 | 通过 |
| 关联测试目录 28-Func-28 存在 | third/28-Func-28 | 通过 |
| 关联测试目录 38-Func-38 存在 | third/38-Func-38 | 通过 |
| 关联测试目录 49-Perf-8 存在 | third/49-Perf-8 | 通过 |
| 关联测试目录 61-Perf-20 存在 | third/61-Perf-20 | 通过 |
| 关联测试目录 71-Perf-30 存在 | third/71-Perf-30 | 通过 |
| 关联测试目录 127-Perf-48 存在 | third/127-Perf-48 | 通过 |
| 关联测试目录 137-Perf-58 存在 | third/137-Perf-58 | 通过 |

## 结果文件

- ../data/function_test.json
