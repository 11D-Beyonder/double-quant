# 83 量子电路可视化功能测试（Func-50） 测试报告

## 测试对象

- 功能编号：83
- 功能名称：实现量子电路可视化
- 被测模块：`double_quant.algorithm.circuit.visualization`
- 测试文件：`tests/double_quant/circuit/83-quantum_circuit_visualization.py`

## 测试命令

```bash
uv run pytest tests/double_quant/circuit/83-quantum_circuit_visualization.py -s
```

## 测试用例

| 用例 | 验证内容 | 预期结果 |
|---|---|---|
| `test_quantum_circuit_visualization_exports_quantum_shapley_risk_circuit` | 量子 Shapley 风险归因电路图和文本电路导出 | PNG 存在且非空；电路含 5 个量子比特；包含 `state_preparation`、`ucry` 和 4 个 `cry` 门 |

## 测试结论

功能脚本验证了量子金融算法电路的可视化能力。相比普通示例电路，该用例展示了风险归因预言机的真实结构，包括区间态制备、受控旋转和风险贡献加载。

本次未运行 pytest，遵守 `AGENTS.md` 的测试执行限制。
