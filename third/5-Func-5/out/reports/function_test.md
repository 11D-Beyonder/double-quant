# Func-5 去中心化金融管理算法——算法功能测试

## 测试内容

本测试检查 去中心化金融管理算法 的算法功能交付是否完整，包括技术报告、应用封装、量子算法电路组件、baseline 组件和关联实验目录。

## 测试结论

- 测试算法：去中心化金融管理算法
- 我们的方法：SFS+Grover 压缩搜索电路
- baseline：普通 Grover 量子搜索电路
- 是否测试通过：通过

## 检查项

| 检查项 | 路径 | 是否通过 |
|---|---|---|
| 算法技术报告存在 | third/5-Func-5/out/reports/report.md | 通过 |
| 应用算法封装存在 | src/double_quant/application/defi_management.py | 通过 |
| 量子算法电路组件存在 | src/double_quant/algorithm/grover/circuit.py | 通过 |
| 量子 baseline 组件存在 | src/double_quant/algorithm/grover/baseline.py | 通过 |
| 关联测试目录 15-Func-15 存在 | third/15-Func-15 | 通过 |
| 关联测试目录 25-Func-25 存在 | third/25-Func-25 | 通过 |
| 关联测试目录 35-Func-35 存在 | third/35-Func-35 | 通过 |
| 关联测试目录 46-Perf-5 存在 | third/46-Perf-5 | 通过 |
| 关联测试目录 58-Perf-17 存在 | third/58-Perf-17 | 通过 |
| 关联测试目录 68-Perf-27 存在 | third/68-Perf-27 | 通过 |
| 关联测试目录 124-Perf-45 存在 | third/124-Perf-45 | 通过 |
| 关联测试目录 134-Perf-55 存在 | third/134-Perf-55 | 通过 |

## 结果文件

- ../data/function_test.json
