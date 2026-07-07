# 第三方测试拆分目录索引

本目录按 `地图v1.4.docx` 第一列全局测试序号拆分，目录命名为 `序号-Func/Perf-x`。`4-Func-4` 至 `10-Func-10` 为算法技术报告；后续目录为单项测试，每个目录包含自己的 `run_test.py`、`out/data`、`out/figures` 和 `out/reports`。

| 全局序号 | 测试项 | 算法 | 测试内容 | 入口脚本 |
|---:|---|---|---|---|
| 14 | `Func-14` | 动态账本更新算法 | 计算所需操作数测试 | `14-Func-14/run_test.py` |
| 15 | `Func-15` | 去中心化金融管理算法 | 计算所需操作数测试 | `15-Func-15/run_test.py` |
| 16 | `Func-16` | 反欺诈监测算法 | 计算所需操作数测试 | `16-Func-16/run_test.py` |
| 17 | `Func-17` | 支付与结算系统算法 | 计算所需操作数测试 | `17-Func-17/run_test.py` |
| 18 | `Func-18` | 贷款发放决策算法 | 计算所需操作数测试 | `18-Func-18/run_test.py` |
| 19 | `Func-19` | 银行网点布局优化算法 | 计算所需操作数测试 | `19-Func-19/run_test.py` |
| 20 | `Func-20` | 指数追踪算法 | 计算所需操作数测试 | `20-Func-20/run_test.py` |
| 24 | `Func-24` | 动态账本更新算法 | 求解空间大小测试 | `24-Func-24/run_test.py` |
| 25 | `Func-25` | 去中心化金融管理算法 | 求解空间大小测试 | `25-Func-25/run_test.py` |
| 26 | `Func-26` | 反欺诈监测算法 | 求解空间大小测试 | `26-Func-26/run_test.py` |
| 27 | `Func-27` | 支付与结算系统算法 | 求解空间大小测试 | `27-Func-27/run_test.py` |
| 28 | `Func-28` | 贷款发放决策算法 | 求解空间大小测试 | `28-Func-28/run_test.py` |
| 29 | `Func-29` | 银行网点布局优化算法 | 求解空间大小测试 | `29-Func-29/run_test.py` |
| 30 | `Func-30` | 指数追踪算法 | 求解空间大小测试 | `30-Func-30/run_test.py` |
| 34 | `Func-34` | 动态账本更新算法 | 计算精度与量子电路参数之间的函数关系测试 | `34-Func-34/run_test.py` |
| 35 | `Func-35` | 去中心化金融管理算法 | 计算精度与量子电路参数之间的函数关系测试 | `35-Func-35/run_test.py` |
| 36 | `Func-36` | 反欺诈监测算法 | 计算精度与量子电路参数之间的函数关系测试 | `36-Func-36/run_test.py` |
| 37 | `Func-37` | 支付与结算系统算法 | 计算精度与量子电路参数之间的函数关系测试 | `37-Func-37/run_test.py` |
| 38 | `Func-38` | 贷款发放决策算法 | 计算精度与量子电路参数之间的函数关系测试 | `38-Func-38/run_test.py` |
| 39 | `Func-39` | 银行网点布局优化算法 | 计算精度与量子电路参数之间的函数关系测试 | `39-Func-39/run_test.py` |
| 40 | `Func-40` | 指数追踪算法 | 计算精度与量子电路参数之间的函数关系测试 | `40-Func-40/run_test.py` |
| 41 | `Func-41` | 7个算法汇总 | 求解精度与量子复杂度分析优化理论测试 | `41-Func-41/run_test.py` |
| 45 | `Perf-4` | 动态账本更新算法 | 不少于多项式级别加速测试 | `45-Perf-4/run_test.py` |
| 46 | `Perf-5` | 去中心化金融管理算法 | 不少于多项式级别加速测试 | `46-Perf-5/run_test.py` |
| 47 | `Perf-6` | 反欺诈监测算法 | 不少于多项式级别加速测试 | `47-Perf-6/run_test.py` |
| 48 | `Perf-7` | 支付与结算系统算法 | 不少于多项式级别加速测试 | `48-Perf-7/run_test.py` |
| 49 | `Perf-8` | 贷款发放决策算法 | 不少于多项式级别加速测试 | `49-Perf-8/run_test.py` |
| 50 | `Perf-9` | 银行网点布局优化算法 | 不少于多项式级别加速测试 | `50-Perf-9/run_test.py` |
| 51 | `Perf-10` | 指数追踪算法 | 不少于多项式级别加速测试 | `51-Perf-10/run_test.py` |
| 57 | `Perf-16` | 动态账本更新算法 | 相较于IBM Qiskit工具精度提升40%及以上测试 | `57-Perf-16/run_test.py` |
| 58 | `Perf-17` | 去中心化金融管理算法 | 相较于IBM Qiskit工具精度提升40%及以上测试 | `58-Perf-17/run_test.py` |
| 59 | `Perf-18` | 反欺诈监测算法 | 相较于IBM Qiskit工具精度提升40%及以上测试 | `59-Perf-18/run_test.py` |
| 60 | `Perf-19` | 支付与结算系统算法 | 相较于IBM Qiskit工具精度提升40%及以上测试 | `60-Perf-19/run_test.py` |
| 61 | `Perf-20` | 贷款发放决策算法 | 相较于IBM Qiskit工具精度提升40%及以上测试 | `61-Perf-20/run_test.py` |
| 62 | `Perf-21` | 银行网点布局优化算法 | 相较于IBM Qiskit工具精度提升40%及以上测试 | `62-Perf-21/run_test.py` |
| 63 | `Perf-22` | 指数追踪算法 | 相较于IBM Qiskit工具精度提升40%及以上测试 | `63-Perf-22/run_test.py` |
| 67 | `Perf-26` | 动态账本更新算法 | 相较于IBM Qiskit工具复杂度降低50%及以上测试 | `67-Perf-26/run_test.py` |
| 68 | `Perf-27` | 去中心化金融管理算法 | 相较于IBM Qiskit工具复杂度降低50%及以上测试 | `68-Perf-27/run_test.py` |
| 69 | `Perf-28` | 反欺诈监测算法 | 相较于IBM Qiskit工具复杂度降低50%及以上测试 | `69-Perf-28/run_test.py` |
| 70 | `Perf-29` | 支付与结算系统算法 | 相较于IBM Qiskit工具复杂度降低50%及以上测试 | `70-Perf-29/run_test.py` |
| 71 | `Perf-30` | 贷款发放决策算法 | 相较于IBM Qiskit工具复杂度降低50%及以上测试 | `71-Perf-30/run_test.py` |
| 72 | `Perf-31` | 银行网点布局优化算法 | 相较于IBM Qiskit工具复杂度降低50%及以上测试 | `72-Perf-31/run_test.py` |
| 73 | `Perf-32` | 指数追踪算法 | 相较于IBM Qiskit工具复杂度降低50%及以上测试 | `73-Perf-32/run_test.py` |
| 123 | `Perf-44` | 动态账本更新算法 | 相较于IBM Qiskit部署方案降低40%以上计算误差测试 | `123-Perf-44/run_test.py` |
| 124 | `Perf-45` | 去中心化金融管理算法 | 相较于IBM Qiskit部署方案降低40%以上计算误差测试 | `124-Perf-45/run_test.py` |
| 125 | `Perf-46` | 反欺诈监测算法 | 相较于IBM Qiskit部署方案降低40%以上计算误差测试 | `125-Perf-46/run_test.py` |
| 126 | `Perf-47` | 支付与结算系统算法 | 相较于IBM Qiskit部署方案降低40%以上计算误差测试 | `126-Perf-47/run_test.py` |
| 127 | `Perf-48` | 贷款发放决策算法 | 相较于IBM Qiskit部署方案降低40%以上计算误差测试 | `127-Perf-48/run_test.py` |
| 128 | `Perf-49` | 银行网点布局优化算法 | 相较于IBM Qiskit部署方案降低40%以上计算误差测试 | `128-Perf-49/run_test.py` |
| 129 | `Perf-50` | 指数追踪算法 | 相较于IBM Qiskit部署方案降低40%以上计算误差测试 | `129-Perf-50/run_test.py` |
| 133 | `Perf-54` | 动态账本更新算法 | 相较于IBM Qiskit部署方案降低50%以上量子计算复杂度测试 | `133-Perf-54/run_test.py` |
| 134 | `Perf-55` | 去中心化金融管理算法 | 相较于IBM Qiskit部署方案降低50%以上量子计算复杂度测试 | `134-Perf-55/run_test.py` |
| 135 | `Perf-56` | 反欺诈监测算法 | 相较于IBM Qiskit部署方案降低50%以上量子计算复杂度测试 | `135-Perf-56/run_test.py` |
| 136 | `Perf-57` | 支付与结算系统算法 | 相较于IBM Qiskit部署方案降低50%以上量子计算复杂度测试 | `136-Perf-57/run_test.py` |
| 137 | `Perf-58` | 贷款发放决策算法 | 相较于IBM Qiskit部署方案降低50%以上量子计算复杂度测试 | `137-Perf-58/run_test.py` |
| 138 | `Perf-59` | 银行网点布局优化算法 | 相较于IBM Qiskit部署方案降低50%以上量子计算复杂度测试 | `138-Perf-59/run_test.py` |
| 139 | `Perf-60` | 指数追踪算法 | 相较于IBM Qiskit部署方案降低50%以上量子计算复杂度测试 | `139-Perf-60/run_test.py` |
