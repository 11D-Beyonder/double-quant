# 86 量子程序智能拼接技术测试（Func-53） 测试报告

## 测试对象

- 功能编号：86
- 功能名称：量子程序智能拼接技术
- 被测模块：`double_quant.algorithm.circuit.stitch`
- 测试文件：`tests/double_quant/circuit/86-circuit_stitch.py`

## 测试命令

```bash
uv run pytest -s tests/double_quant/circuit/86-circuit_stitch.py
```

## 测试用例

| 用例 | 验证内容 | 预期结果 |
|---|---|---|
| `test_stitch_same_width_circuits_matches_manual_compose` | 两段同线宽电路自动拼接 | 输出电路与手动 `compose()` 的状态向量等价 |
| `test_stitch_with_explicit_qubit_map_matches_manual_circuit` | 显式指定 `qubit_map={0: 1, 1: 0}`，验证右侧 `q[0]` 接入输出 `q[1]`、右侧 `q[1]` 接入输出 `q[0]` | 拼接后 CNOT 控制点位于 `q_1`，目标 X 位于 `q_0`，且输出电路与手动拼接状态向量等价 |
| `test_stitch_extends_left_circuit_when_allowed` | 允许自动扩展左侧电路线宽 | 输出电路线宽扩展到右侧需求，并记录 `EXTENDED_QUBITS` |
| `test_stitch_rejects_incompatible_width_without_extend` | 左侧 1 个量子比特含 `H` 门，右侧 2 个量子比特含 `CX` 门，且 `allow_extend=False` | 拒绝从 1 线宽隐式扩展到 2 线宽，抛出 `CircuitStitchingError` 并输出拒绝原因 |

## 实测结果

```text
4 个用例通过，耗时 1.28 秒
```

## 测试结论

量子程序智能拼接功能通过单文件测试。测试输出展示了拼接前左侧电路、右侧电路、拼接后电路、量子比特映射和 经典比特映射，能够直接证明功能点完成了电路级智能拼接。
