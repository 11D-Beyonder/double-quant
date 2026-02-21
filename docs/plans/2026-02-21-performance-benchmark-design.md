# 量子风险归因算法性能基准实验设计

## 概述

本设计文档描述了量子风险归因算法的性能评估实验方案，包括量子方法间的横向对比以及量子算法与经典蒙特卡洛方法的效率对比。

## 目标

1. **量子方法横向对比**：在固定 interval qubits 下，比较所有量子提取方法的精度
2. **量子 vs 经典对比**：比较最优量子方法与经典 MC 在相近 oracle 调用次数下的精度

## 实验设计

### 第一阶段：量子方法横向对比

**目标**：在固定 interval qubits 下，比较所有量子提取方法的精度

**实验参数**：

| 变量 | 值 |
|------|-----|
| 玩家数 (n) | 3, 4, 5, 6 |
| Interval qubits (n_l) | 3, 4, 5, 6, 7 |
| 提取方法 | statevector, shots(1024), shots(4096), qae_iqae, qae_mlqae, qae_fae |
| MC 轮数 | 50 |
| Ground truth | `BinaryEnumerationCalculator` |
| 度量 | 平均相对误差 |

**输出**：
- 每个 n 值一张折线图（x=n_l, y=mean_rel_err，不同方法用不同颜色/线型）
- 汇总表格：各方法在 n_l=7 时的平均误差

### 第二阶段：最优量子 vs 经典 MC

**目标**：比较最优量子方法与经典 MC 在相近 oracle 调用次数下的精度

**经典 MC 实现**：
- 方法：排列采样法（Permutation Sampling）
- 公式：φ̂_i = (1/T) Σ_{t=1}^T [v(P_i^t ∪ {i}) - v(P_i^t)]
- 采样次数 T = 10, 20, 50, 100
- Oracle 调用次数 = T × n

**输出**：
- 对比图：oracle 调用次数 vs 平均相对误差
- 汇总：达到相同精度时，量子 vs 经典的 oracle 调用次数比率

## 代码变更

### 1. `src/double_quant/solver/shapley.py`

**新增 extraction_mode**：
```python
ExtractionMode = Literal[
    "statevector", "shots",
    "qae_canonical", "qae_iqae", "qae_mlqae", "qae_fae"  # 新增 qae_fae
]
```

**QAEOptions 新增字段**：
```python
@dataclass
class QAEOptions:
    shots: int = 1024
    epsilon: float = 0.01
    alpha: float = 0.05
    num_eval_qubits: int = 3
    # FAE 专用参数
    delta: float = 0.05      # FAE: 置信参数
    maxiter: int = 5         # FAE: 最大迭代次数
```

**_run_qae 方法新增分支**：
```python
from qiskit_algorithms import FasterAmplitudeEstimation

# 在 _run_qae 方法中新增：
elif self._extraction_mode == "qae_fae":
    algo = FasterAmplitudeEstimation(
        delta=opts.delta,
        maxiter=opts.maxiter,
        sampler=sampler,
    )
```

**新增 PermutationMCCalculator 类**：
```python
class PermutationMCCalculator(ShapleyCalculator):
    """使用排列采样法的经典蒙特卡洛 Shapley 估计器。"""

    def __init__(
        self,
        num_players: int,
        value_dict: ValueFunction | None = None,
        num_samples: int = 100,
        seed: int | None = None,
    ):
        super().__init__(num_players, value_dict)
        self.num_samples = num_samples
        self.rng = np.random.default_rng(seed)

    def _calculate_one(self, target_player: int) -> float:
        # 排列采样法实现
        ...
```

### 2. `tests/double_quant/application/test_risk.py`

**新增测试类**：
```python
class TestPerformanceBenchmark:
    def test_quantum_methods_comparison(self):
        """第一阶段：量子方法横向对比（折线图）"""

    def test_quantum_vs_classical_mc(self):
        """第二阶段：最优量子 vs 经典 MC"""
```

## 可视化规范

- 使用 seaborn whitegrid 主题
- 字体：Times New Roman
- 折线图：不同方法用不同颜色/线型区分
- 输出格式：SVG

## 参考文献

- 排列采样法：参考 `tests/double_quant/application/data/mc.pdf`
- FasterAmplitudeEstimation: [Nakaji, 2020] arXiv:2002.02417
