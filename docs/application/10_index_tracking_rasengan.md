# 算法10：指数追踪算法（Rasengan 离散成分股选择）

## 1. 算法定位

指数追踪算法被建模为带行业/因子约束的离散成分股选择问题。变量表示是否选择候选资产，目标为最小化组合因子暴露与目标指数之间的跟踪误差，并加入交易或持有成本。

## 2. 数学形式

定义二元变量

$$
x_i \in \{0,1\}.
$$

`x_i=1` 表示选择第 `i` 只候选资产。设资产因子暴露为 `a_i`，目标指数暴露为 `b`，成本为 `c_i`。目标函数为：

$$
\min_x f(x)
= \lVert A x - b\rVert_2^2 + \sum_i c_i x_i.
$$

行业或分组约束为：

$$
\sum_{i\in S_m}x_i = 1,\qquad m=1,\ldots,M.
$$

验证样例包含科技、金融、消费三个行业组，每组选择一只资产。

## 3. 求解方法

采用 Rasengan。该问题是典型约束二元优化：目标为二次函数，约束为分组 one-hot。transition move 为行业内资产替换，天然保持每个行业组的选择数量不变。

## 4. 具体实现

实现流程：

1. 计算候选资产因子暴露矩阵 `A` 与目标指数暴露 `b`。
2. 设定行业/规模/流动性等分组约束。
3. 构造初始行业均衡组合。
4. 构造行业内 swap transition moves。
5. Rasengan 在可行组合空间中搜索。
6. 按 tracking error 与成本输出最优组合。

## 5. Baseline 与优势口径

Baseline 为 Low 的 QAOA。Rasengan 避免 penalty-QAOA 对行业约束的软惩罚问题，采样态始终处于行业均衡可行空间，适合 NISQ 场景下的离散指数追踪。

## 6. 验证结果

临时实验中：

```text
可行解数量 = 8 / 64
最优可行解 = 100101
目标值 = 0.0785
Rasengan moves = 3, 覆盖 8/8
SFS-one-hot-product moves = 3, 覆盖 8/8
Penalty-QAOA 可行概率 = 0.9845
```

对应代码与报告见 `temp/rasengan_sfs_binary_opt`。
