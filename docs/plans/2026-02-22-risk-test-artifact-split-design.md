# Risk 测试与出图解耦设计

## 背景与问题

当前 `tests/double_quant/application/test_risk.py` 同时承担了三类职责：

1. 正确性断言（应由 pytest 负责）
2. 重计算实验（数据下载、量子/经典多轮基准）
3. 论文图像生成（样式迭代频繁）

这导致一个直接问题：仅想调整图像样式时，也需要重跑高成本计算。

## 目标

1. 将 correctness 测试与实验/出图流程彻底分离
2. 保留图片版本管理，便于文档引用与追溯
3. 支持“改样式只重画图，不重算数据”

## 非目标

1. 不改变风险归因核心数学定义与求解逻辑
2. 不新增外部依赖
3. 不在本次设计中重写已有实验方法学

## 方案总览（已确认）

采用“双阶段脚本 + 产物分层”方案：

- 阶段 A：`generate_artifacts` 负责重计算并产出结构化中间数据
- 阶段 B：`plot_from_artifacts` 只读取中间数据并生成最终图像
- pytest 仅保留快速、可重复的断言测试

## 拆分范围与处理策略

### 保留在 pytest 的测试

- `test_permutation_mc_basic`
- `TestRiskSaving.test_superadditivity`
- `TestRiskSaving.test_restoration_accuracy`（仅保留数值断言，移除绘图段）
- `TestQuantumSolver` 下小规模正确性测试

### 从 pytest 移出的逻辑

- `test_volatility_bucketing` 中指标计算与 `savefig` 段
- `test_restoration_accuracy` 中横向条形图绘图段
- `TestQuantumPerformance` 中重实验逻辑移入脚本工作流

### 明确删除

- 直接删除 `test_quantum_vs_classical_mc` 测试代码

## 目录与版本管理策略

### 缓存层（不进 git）

- 路径：`tests/double_quant/application/cache/`
- 内容：下载缓存、重计算临时产物等

### 快照数据层（进 git）

- 路径：`docs/assets/risk/data/`
- 内容：出图所需轻量 CSV（聚合/序列化后的稳定数据）

### 最终图层（进 git）

- 路径：`docs/assets/risk/`
- 内容：论文/文档引用的 PNG 图

### 忽略规则

- 更新 `tests/double_quant/application/.gitignore`，忽略 `cache`
- 为兼容历史产物，可同时保留对 `data` 的忽略

## 脚本入口与命令

- 产数据：`uv run scripts/risk/generate_artifacts.py`
- 出图：`uv run scripts/risk/plot_from_artifacts.py`

约束：

- `plot_from_artifacts.py` 不触发重计算
- `generate_artifacts.py` 不直接输出最终图（只产数据）

## 中间数据建议格式

建议在 `docs/assets/risk/data/` 维护以下快照数据：

1. `vol_buckets_metrics.csv`
2. `vol_buckets_series.csv`
3. `restoration_accuracy.csv`
4. `quantum_comparison_n{n}.csv`
5. `equal_error_oracle_calls_summary.csv`
6. `manifest.json`（记录参数、随机种子、时间戳、输入路径）

## 文档落位

不将实验命令写入 `README.md`，新增专门文档：

- `docs/application/risk-experiment-workflow.md`

文档内容覆盖：目录约定、两步命令、常见失败场景与重跑说明。

## 错误处理与可维护性

1. `plot_from_artifacts.py` 检测输入快照不存在时，给出明确提示并退出
2. `generate_artifacts.py` 提供 `--force` 支持强制重算
3. 生成流程每次更新 `manifest.json`，确保图与数据可追溯

## 风险与缓解

1. **风险：** 快照数据与代码逻辑漂移  
   **缓解：** 通过 `manifest.json` 固定参数与生成版本信息
2. **风险：** 用户误将缓存层作为文档数据源  
   **缓解：** 在 workflow 文档中明确缓存层仅用于临时计算
3. **风险：** pytest 仍被慢逻辑拖慢  
   **缓解：** 移除重实验与出图代码，保持测试纯断言

## 验收标准

1. 调整绘图样式时，仅运行 `uv run scripts/risk/plot_from_artifacts.py` 即可完成更新
2. pytest 默认执行不再触发图片生成
3. `docs/assets/risk/` 中图像可直接被文档引用且可版本回溯
4. `test_quantum_vs_classical_mc` 已从测试文件中删除
