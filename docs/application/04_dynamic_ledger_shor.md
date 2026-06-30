# 算法4：动态账本更新算法（Shor 周期查找/因式分解）

## 1. 算法定位

动态账本更新算法面向账本安全参数、承诺参数或加密模数更新中的周期查找与因式分解子问题。该应用不建模为约束二元优化，而作为 Shor 类量子算法应用：当账本系统需要识别或更新合数模数 `N` 的安全结构时，通过量子周期查找恢复 `N=pq` 的非平凡因子。

## 2. 数学形式

给定合数模数

$$
N = pq,\qquad p,q \text{ 为未知非平凡因子}
$$

选择与 `N` 互素的底数 `a`，寻找最小正整数 `r` 使

$$
a^r \equiv 1 \pmod N
$$

若 `r` 为偶数且 `a^(r/2) != -1 mod N`，则

$$
\begin{aligned}
p &= \gcd(a^{r/2}-1, N),\\
q &= \gcd(a^{r/2}+1, N).
\end{aligned}
$$

## 3. 求解方法

采用 Shor 算法。量子部分负责 order finding，即估计 `a^x mod N` 的周期 `r`；经典后处理用最大公约数恢复因子。测试实现中用小规模 `N=91, a=3` 验证后处理流程：

$$
\begin{aligned}
3^6 &\equiv 1 \pmod{91},\\
\gcd(3^3-1,91) &= 13,\\
\gcd(3^3+1,91) &= 7.
\end{aligned}
$$

## 4. 具体实现

实现步骤如下：

1. 输入动态账本安全模数 `N`。
2. 选择底数 `a`，若 `gcd(a,N)>1` 则直接得到因子。
3. 调用 Shor order-finding 子程序得到周期 `r`。
4. 判断 `r` 是否满足偶数和非平凡条件。
5. 计算 `gcd(a^(r/2)-1,N)` 与 `gcd(a^(r/2)+1,N)`。
6. 输出账本更新所需的因子信息或安全告警。

## 5. Baseline 与优势口径

Baseline 为经典试除、经典 NP/亚指数因式分解流程。小规模验证中，Shor 子程序视为一次 order-finding 查询，经典试除需检查多个候选除数。大规模口径下，Shor 具有多项式时间因式分解优势，而经典通用因式分解没有已知多项式时间算法。

## 6. 验证结果

临时实验结果：

```text
N = 91
a = 3
r = 6
factors = 7 x 13
quantum order-finding query = 1
classical trial candidates = 6
```

对应代码与报告见 `temp/shor_grover_remaining`。
