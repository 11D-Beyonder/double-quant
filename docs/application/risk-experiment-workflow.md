# Risk Experiment Workflow

## 目录约定

- 缓存层（不进 git）：`tests/double_quant/application/cache/`
- 快照数据层（进 git）：`docs/assets/risk/data/`
- 最终图片层（进 git）：`docs/assets/risk/`

## 运行顺序

如果网络环境受限，先设置代理：

```bash
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 all_proxy=socks5://127.0.0.1:7897
```

1. 生成快照数据（重计算阶段）

```bash
uv run scripts/risk/generate_artifacts.py
```

2. 基于快照数据出图（样式迭代阶段）

```bash
uv run scripts/risk/plot_from_artifacts.py
```

## 常见场景

- 只改图像样式：只执行 `uv run scripts/risk/plot_from_artifacts.py`
- 需要重算数据：执行 `uv run scripts/risk/generate_artifacts.py --force`

## 错误处理

- 若绘图脚本提示快照缺失，请先运行生成脚本。
- 每次生成会更新 `docs/assets/risk/data/manifest.json`，用于追踪参数与来源数据。
