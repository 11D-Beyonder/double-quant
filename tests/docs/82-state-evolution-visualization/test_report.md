# 82 量子态演化可视化功能测试（Func-49） 测试报告

## 测试对象

- 功能编号：82
- 功能名称：实现量子态演化可视化
- 被测模块：`double_quant.algorithm.circuit.visualization`
- 测试文件：`tests/double_quant/circuit/82-quantum_state_evolution_visualization.py`

## 测试命令

```bash
uv run pytest tests/double_quant/circuit/82-quantum_state_evolution_visualization.py -s
```

## 测试用例

| 用例 | 验证内容 | 预期结果 |
|---|---|---|
| `test_state_evolution_visualizes_quantum_shapley_risk_oracle_as_gif` | 量子 Shapley 风险归因预言机的态演化、布洛赫球、概率分布和 GIF 导出 | PNG 与 GIF 均存在且非空；演化步数等于电路门数加一；最终概率归一化为 1；输出量子比特振幅概率大于 0 |

## 实测结论

功能脚本不再使用 Bell 态或随机电路，而是使用风险归因场景下的 量子 Shapley 电路。可视化结果覆盖：

- 基态概率分布柱状图
- 单量子比特 布洛赫球
- 逐门状态演化 GIF
- 风险贡献归一化因子和 输出量子比特 概率输出

本次未运行 pytest，遵守 `AGENTS.md` 的测试执行限制。
