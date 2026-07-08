# 75 完成数据获取、分析、编解码流程测试（Func-43） 测试报告

## 测试对象

- 功能编号：75
- 功能名称：完成数据获取、分析、编解码的流程
- 被测模块：
  - `double_quant.data.source`
  - `double_quant.data.transform`
  - `double_quant.application.risk`
  - `double_quant.algorithm.shapley`
- 测试文件：`tests/double_quant/data/75-data_analysis_encoding_flow.py`

## 测试命令

```bash
uv run pytest tests/double_quant/data/75-data_analysis_encoding_flow.py -s
```

## 测试用例

| 用例 | 验证内容 | 预期结果 |
|---|---|---|
| `test_data_fetch_analysis_encoding_flow` | 验证从数据获取到金融分析，再到金融集合和量子振幅编解码的完整流程 | 输出价格矩阵、收益率、协方差、期望收益、Shapley 风险归因，并成功完成 bitmask 与量子振幅读回 |

## 实测结果

```text
1 passed, 2 warnings in 1.80s
```

## 结果分析

本功能测试覆盖拆分后的第二个流程型功能。

第一阶段为数据获取。测试通过 `YFinanceSource.fetch()` 的统一接口获得价格矩阵。为保证验收测试稳定，外部下载函数由 monkeypatch 替身数据提供，不依赖真实网络。

第二阶段为数据分析。测试将价格矩阵输入：

- `to_log_returns()`
- `to_covariance()`
- `to_expected_returns()`
- `RiskAttributor(...)`

输出 log 收益率、协方差矩阵、期望收益和 Shapley ES 风险归因。

第三阶段为编解码。测试先将资产集合编码为 bitmask，再解码为资产名称列表；随后将 ES 风险值归一化并输入 `ValueLoader`，由量子线路把风险值编码为输出比特振幅概率，最后通过 `Statevector.probabilities()` 读回并恢复原始 ES 值。

## 测试结论

功能 75 通过单文件验收测试。测试证明当前仓库已经形成：

```text
数据获取 -> 金融分析 -> 金融集合编码 -> 量子振幅编码 -> 概率读回解码
```

的完整流程。

测试中的 warning 来自 Qiskit 对 `BlueprintCircuit` 的弃用提示，属于既有 `ValueLoader` 量子线路实现依赖的上游 API 提醒，不影响本功能结论。

