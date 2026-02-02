# 定义

定义合作博弈 $G=(F,V)$，$F=\{0,1,\cdots,n\}$ 是 $n+1$ 个玩家的集合，$V : \mathcal{P}(F)\to\mathbb{R}$ 作为收益函数，$V(S)$ 是指一个联盟的合作收益，其中 $S\subseteq F$、$V(\varnothing)=0$。

$$
\Phi_i=\sum_{S\subseteq F-\{i\}} \gamma(|F-\{i\}|,|S|) [V(S\cup\{i\})-V(S)]
$$

定义 $\gamma$ 为权重函数：

$$
\gamma(n,m)=\frac{m!(n-m)!}{(n+1)!}
$$

==枚举不包含 $i$ 的子集 $S$ 求和，权重和集合大小$|S|$相关。==

这里只考虑**超加和性博弈**，即加入新成员后的联盟收益必定不减少：

$$
\forall S,H\subseteq F,\quad V(S\cup H)\ge V(S)+V(H)
$$


定义一个算子（可证明是酉矩阵）：

$$
B=I^{\otimes n}\otimes\begin{bmatrix}\sqrt{1-\phi(x,n)}&\sqrt{\phi(x,n)}\\\sqrt{\phi(x,n)}&-\sqrt{1-\phi(x,n)}\end{bmatrix}
$$

其中 $\phi(x,n)=\gamma(n,x\text{中1的个数})\cdot\hat{W}(x)$。则最终量子态要落到：

$$
B(H^{\otimes n}\otimes I)|0\rangle^{\otimes{n+1}}=\sum_{x=0}^{2^n-1}\frac{1}{\sqrt{2^n}}|x\rangle\left(\sqrt{1-\phi(x,n)}|0\rangle+\sqrt{\phi(x,n)}|1\rangle\right)
$$

测量辅助比特（第$n+1$个），期望值为

$$
\sum_{x=0}^{2^n-1}\frac{1}{2^n}\phi(x,n)=\frac{1}{2^n W_{\max}}\sum_{S\subseteq F-\{i\}} \gamma(|F-\{i\}|,|S|) [V(S\cup\{i\})-V(S)]
$$

**需要重复制备量子态$O(2^n)$次。** 


# 量子算法

## 思路

为了方便说明，定义玩家 $i$ 加入联盟 $S$ 后产生的贡献为

$$
W(S)=V(S\cup\{i\})-V(S),\quad S\subseteq F-\{i\}
$$

**超加和性博弈**的限制保证了 $W(S)\ge 0$。根据计算规则

$$
\Phi_i=\sum_{S\subseteq F-\{i\}} \gamma(|F-\{i\}|,|S|) W(S)
$$

### 二进制表示

**考查 $i$ 时，用 $n$ 个比特位 $h=\overline{h_0h_1\cdots h_{i-1}h_{i+1}\cdots h_n}$ 表示其余玩家的状态，也就是集合 $S$。**

==计算规则转变为：枚举 $h\in\{|00\cdots 0\rangle,|00\cdots 1\rangle,|11\cdots 1\rangle\}$ 求和，权重和$h$中$1$的个数相关。==

### 权重近似

**用黎曼和表示的定积分 $\beta_{n,m}$ 近似 $\gamma$。**
定义贝塔函数：

$$
\beta_{n,m}=\int_0^1 x^m(1-x)^{n-m}\text{d}x,\quad 0\le m\le n,\quad m,n\in\mathbb{N}
$$

可以证明 $\beta_{n,m}=\gamma(n,m)$，因此不妨用黎曼和来求积分近似值。

> **证明见正文。**

### 计算 $W(S)$

**用一个旋转门 $U_W$ 实现 $W(S)$ 计算。**

$$
U_W|h\rangle|0\rangle=|h\rangle\left(\sqrt{1-\hat{W}(h)}|0\rangle+\sqrt{\hat{W}(h)}|1\rangle\right)
$$

这里使用的是 $\hat{W}(h)$，这是一个被放缩到 $[0,1]$ 的值。已知 $i$ 能产生的最大贡献为

$$
W_{\max}\ge\max_{S\subseteq F-\{i\}} W(S)
$$

显然用 $h$ 可以表示一个联盟 $S_h$，即 $h$ 中二进制位 $h_j=1$ 表示 $j$ 在联盟 $S_h$ 中，接着可定义：

$$
\hat{W}(h)=\frac{W(S_h)}{W_{\max}}
$$

## 量子态演化

初始量子态：

$$
|\psi_0\rangle=|0\rangle_{\text{Pt}}\otimes |0\rangle_{\text{Pl}}\otimes |0\rangle_{\text{Ut}}
$$

- 区间寄存器（Pt），存储近似 $\beta_{n,m}$ 的各个区间。
- 玩家寄存器（Pl），用 $n$ 比特表示前面提到的联盟 $S_h$，利用叠加性可表示 $2^n$ 个联盟。
- 收益寄存器（Ut），用单个比特对收益进行编码，即 $U_W|h\rangle|0\rangle=|h\rangle\left(\sqrt{1-\hat{W}(h)}|0\rangle+\sqrt{\hat{W}(h)}|1\rangle\right)$。

考虑区间 $[0,1]$ 中的 $2^l+1$ 个点 $P_{l}=\{t_l(k)\}_{k=0}^{2^l}$，其中

$$
t_l(k)=\sin^2\left(\frac{\pi k}{2^{l+1}}\right)
$$
可以用 $2^l$ 个区间 $w_l(k)=t_l(k+1)-t_l(k)$ 进行黎曼和近似。我们可以将 Pt 制备成 $\sum_{k=0}^{2^l-1}\sqrt{w_l(k)}|k\rangle$，系统的量子态转变为 

$$
|\psi_1\rangle=\sum_{k=0}^{2^l-1} \sqrt{w_l(k)}|k\rangle_{\text{Pt}}|0\rangle_{\text{Pl}}|0\rangle_{Ut}
$$


下面引入一个电路 $R$，满足

$$
R|k\rangle|0\rangle=|k\rangle\left(\sqrt{1-t^\prime_l(k)}|0\rangle+\sqrt{t^\prime_l(k)}|1\rangle\right)
$$

其中 $t_l^\prime(k)=t_{l+1}(2k+1)$，是区间 $[t_l(k),t_l(k+1)]$ 内的点。

> **证明：$t_{l+1}(2k+1)\in [t_l(k),t_l(k+1)]$**
> 
> - $t_l(k)=\sin^2\left(\frac{\pi}{2}\cdot \frac{k}{2^l}\right)$
> - $t_{l+1}(2k+1)=\sin^2\left[\frac{\pi}{2}\cdot\left(\frac{k}{2^l}+\frac{1}{2^{l+1}}\right)\right]$
> - $t_{l}(k+1)=\sin^2\left[\frac{\pi}{2}\cdot\left(\frac{k}{2^l}+\frac{1}{2^{l}}\right)\right]$
> 
> $\sin^2(\cdot)$ 括号内的值属于 $\left[0,\frac{\pi}{2}\right]$，且逐渐递增。$y=\sin^2 x$ 在 $x\in[0,\frac{\pi}{2}]$ 单调递增，得证。

量子电路如下：

![draw.svg](https://picgo-1306543186.cos.ap-chongqing.myqcloud.com/202407051928605.svg)


> **证明：上图电路就是 $R$**
> 
> 每个受控门的旋转角依次为 $\frac{\pi}{2},\cdots,\frac{\pi}{2^{l-1}}$，若 $k$ 中某位为 $1$ 则旋转该角度，否则什么都不做。因此可以抽象成一系列普通旋转门，角度依次为 $\frac{\pi}{2}k_{l-1},\frac{\pi}{4}k_{l-2},\cdots,\frac{\pi}{2^{l}}k_0,\frac{\pi}{2^{l+1}}$。所有角度加起来作用 $R_Y$ 门，得到
> 
 > $$
> \begin{aligned}
> R&=R_Y\left(\frac{\pi}{2^{l+1}}+\sum_{i=0}^{l-1}\frac{\pi}{2^{l-i}}k_{i}\right)\\
> &=R_Y\left(\frac{\pi}{2^{l+1}}+\frac{\pi}{2^l}\sum_{i=0}^{l-1}2^ik_{i}\right)\\
> &=R_Y\left(\frac{\pi}{2^{l+1}}+\frac{\pi}{2^l}k\right)\\
> &=R_Y\left[\pi\cdot \left(\frac{k}{2^l}+\frac{1}{2^{l+1}}\right)\right]
> \end{aligned}
> $$
>
> 代入 $R|x\rangle|0\rangle$，再结合 $R_Y(\theta)|0\rangle=\cos\left(\frac{\theta}{2}\right)|0\rangle+\sin\left(\frac{\theta}{2}\right)|1\rangle$，得证。

对Pl中每个比特作用此电路，得到

$$
|\psi_2\rangle=\sum_{k=0}^{2^l-1}\sqrt{w_l(k)}|k\rangle_{\text{Pt}}\left(\sqrt{1-t^\prime_l(k)}|0\rangle+\sqrt{t^\prime_l(k)}|1\rangle\right)^{\otimes n}_{\text{Pl}}|0\rangle_{\text{Ut}}
$$

令 $H_m$ 为与 $|0\rangle^n$ 汉明距离为 $m$ 的 $n$ 比特量子态集合（其实就是数 $1$ 的个数）。从汉明距离的角度看，此时Pl中的量子态具有特殊性质。对于 $n$ 比特的Pl寄存器来说，量子态可以写为

$$
\sum_{m=0}^n\sqrt{\left(t^{\prime}_{l}(k)\right)^m\left(1-t^\prime_l(k)\right)^{n-m}}\sum_{h\in H_m}|h\rangle
$$

> 以 $n=2$ 举例，我们有
>
> $$
> \left(\sqrt{1-t^\prime_l(k)}|0\rangle+\sqrt{t^\prime_l(k)}|1\rangle\right)^{\otimes 2}=\sqrt{\left(1-t^\prime_l(k)\right)^2}|00\rangle+\sqrt{t^\prime_l(k)\left(1-t^\prime_l(k)\right)}|01\rangle+\sqrt{t^\prime_l(k)\left(1- t^\prime_l(k)\right)}|10\rangle+\sqrt{\left(t_l^\prime(k)\right)^2}|11\rangle
 >$$
> 
> 即 $H_0=\{|00\rangle\}$、$H_1=\{|01\rangle,|10\rangle\}$、$H_2=\{|11\rangle\}$。于是上述量子态可以写为
> 
> $$
> \sqrt{\left(1-t^\prime_l(k)\right)^2}\sum_{h\in H_0}|h\rangle+\sqrt{t^\prime_l(k)\left(1-t^\prime_l(k)\right)}\sum_{h\in H_1}|h\rangle+\sqrt{\left(t_l^\prime(k)\right)^2}\sum_{h\in H_2}|h\rangle
> $$

用上述形式描述 $|\psi_2\rangle$，得到

$$
|\psi_2\rangle=\sum_{k=0}^{2^l-1}\sum_{m=0}^n\sum_{h\in H_m}\sqrt{w_l(k)\left(t^{\prime}_{l}(k)\right)^m\left(1-t^\prime_l(k)\right)^{n-m}}|k\rangle_{\text{Pt}}|h\rangle_{\text{Pl}}|0\rangle_{\text{Ut}}
$$

交换下求和符号，最内侧枚举 $k$ 的求和得到的是 $\beta_{n,m}$ 的近似值：

$$
|\psi_2\rangle=\sum_{m=0}^n\sum_{h\in H_m}\sum_{k=0}^{2^l-1}\sqrt{w_l(k)\left(t^{\prime}_{l}(k)\right)^m\left(1-t^\prime_l(k)\right)^{n-m}}|k\rangle_{\text{Pt}}|h\rangle_{\text{Pl}}|0\rangle_{\text{Ut}}
$$

> 为什么这里交换求和符号是成立的？简单来说是因为 $k,m,h$ 三者相互独立。详见[《具体数学》2.4节关于多重和式介绍](https://www.bilibili.com/video/BV1ZA411A7St)。

### $U_W$ 门

记 $|W(h)\rangle=\sqrt{1-\hat{W}(h)}|0\rangle+\sqrt{\hat{W}(h)}|1\rangle$，那么对 Ut 作用 $U_W$ 门后，得到量子态：

$$
|\psi_3\rangle=\sum_{m=0}^n\sum_{h\in H_m}\sum_{k=0}^{2^l-1}\sqrt{w_l(k)\left(t^{\prime}_{l}(k)\right)^m\left(1-t^\prime_l(k)\right)^{n-m}}|k\rangle_{\text{Pt}}|h\rangle_{\text{Pl}}|W(h)\rangle_{\text{Ut}}
$$

### 终态分析

可以证明 

$$
\lim_{l\to +\infty}\sum_{k=0}^{2^l-1} w_l(k)\left(t^{\prime}_{l}(k)\right)^m\left(1-t^\prime_l(k)\right)^{n-m}=\gamma(n,m)
$$

> **证明：这是显然的，因为这个式子就是拆分区间去近似定积分。**

因此

$$
\text{tr}_{\text{Pt,Pl}}(|\psi_3\rangle\langle\psi_3|)\approx \sum_{m=0}^n\sum_{h\in H_m}\gamma(n,m)|W(h)\rangle_{\text{Ut}}\langle W(h)|_{\text{Ut}}
$$

于是测量得到 $1$ 的期望为

$$
\sum_{m=0}^n\sum_{h\in H_m}\gamma(n,m)\hat{W}(h)=\frac{1}{W_{\max}}\sum_{m=0}^n\sum_{h\in H_m}\gamma(n,m)(V(S_h\cup \{i\})-V(S_h))
$$

与 $\Phi_i$ 的定义仅有系数 $\frac{1}{W_{\max}}$ 的差别。