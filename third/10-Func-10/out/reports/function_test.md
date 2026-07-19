# Func-10 指数追踪算法——算法功能测试

## 测试内容

本测试检查 指数追踪算法 的算法功能交付是否完整，包括技术报告、应用封装、量子算法电路组件、baseline 组件和关联实验目录。

## 测试结论

- 测试算法：指数追踪算法
- 我们的方法：Rasengan 行业约束篮子搜索电路
- baseline：Penalty-QAOA 电路
- 是否测试通过：通过

## 检查项

| 检查项 | 路径 | 是否通过 |
|---|---|---|
| 算法技术报告存在 | third/10-Func-10/out/reports/report.md | 通过 |
| 应用算法封装存在 | src/double_quant/application/index_tracking.py | 通过 |
| 量子算法电路组件存在 | src/double_quant/algorithm/rasengan/circuit.py | 通过 |
| 量子 baseline 组件存在 | src/double_quant/algorithm/rasengan/baseline.py | 通过 |
| 关联测试目录 20-Func-20 存在 | third/20-Func-20 | 通过 |
| 关联测试目录 30-Func-30 存在 | third/30-Func-30 | 通过 |
| 关联测试目录 40-Func-40 存在 | third/40-Func-40 | 通过 |
| 关联测试目录 51-Perf-10 存在 | third/51-Perf-10 | 通过 |
| 关联测试目录 63-Perf-22 存在 | third/63-Perf-22 | 通过 |
| 关联测试目录 73-Perf-32 存在 | third/73-Perf-32 | 通过 |
| 关联测试目录 129-Perf-50 存在 | third/129-Perf-50 | 通过 |
| 关联测试目录 139-Perf-60 存在 | third/139-Perf-60 | 通过 |

## 结果文件

- ../data/function_test.json
