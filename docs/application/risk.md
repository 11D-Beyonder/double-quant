# Risk Attribution

## 1. Risk Measure: Expected Shortfall

Portfolio risk attribution requires a risk measure that decomposes sensibly across assets. **Value at Risk (VaR)** at confidence level α is the loss threshold not exceeded with probability α:

$$\text{VaR}_\alpha(X) = \inf\{l \in \mathbb{R} : P(L > l) \leq 1 - \alpha\}$$

VaR is widely used but has a critical flaw: it is **not subadditive**, meaning the risk of a combined portfolio can exceed the sum of individual risks, which violates the intuition that diversification reduces risk.

**Expected Shortfall (ES)**, also known as CVaR, is the expected loss conditional on the loss exceeding VaR:

$$\text{ES}_\alpha(X) = \mathbb{E}[L \mid L \geq \text{VaR}_\alpha(X)]$$

ES satisfies all four axioms of a **coherent risk measure** (Artzner et al., 1999): monotonicity, translation invariance, positive homogeneity, and crucially **subadditivity**:

$$\text{ES}(X + Y) \leq \text{ES}(X) + \text{ES}(Y)$$

Subadditivity formalises the benefit of diversification and makes ES suitable as a cooperative game's characteristic function.

> **Implementation:** `double_quant.common.metric.expected_shortfall` computes ES via historical simulation — sorting losses and averaging those beyond the α-quantile.

---

## 2. Cooperative Game Model

Risk attribution is modelled as an **n-player cooperative game** (N, v):

- **Players** N = {1, 2, …, n} correspond to the n assets in the portfolio.
- **Characteristic function** v : 2^N → ℝ assigns a value to every coalition S ⊆ N.

Using ES directly as the characteristic function — V(S) = ES(S) — the **Shapley Risk Contribution (SRC)** of asset i is its Shapley value in this game:

$$\text{SRC}_i = \Phi_i(V) = \sum_{S \subseteq N \setminus \{i\}} \gamma(n, |S|) \left[ \text{ES}(S \cup \{i\}) - \text{ES}(S) \right] \tag{Eq. 4.6}$$

where the Shapley weight is:

$$\gamma(n, s) = \frac{s!\,(n - s - 1)!}{n!}$$

The SRC values satisfy the **efficiency axiom**: $\sum_i \text{SRC}_i = \text{ES}(N)$, meaning the total portfolio risk is exactly decomposed across assets.

> **Implementation:** `RiskAttributor(mode="es")` uses `ExpectedShortfallValueFunction` as the characteristic function and computes SRC = Φ_i^ES directly.

---

## 3. The Subadditivity–Superadditivity Conflict

The quantum Shapley algorithm (see [docs/solver/shapley.md](../solver/shapley.md)) requires the characteristic function to be **superadditive**:

$$v(S \cup T) \geq v(S) + v(T) \quad \forall\, S \cap T = \emptyset$$

Because ES is *subadditive*, V(S) = ES(S) satisfies the *opposite* inequality. Passing ES directly as the characteristic function violates the quantum algorithm's precondition and produces incorrect Shapley estimates.

---

## 4. Risk Saving Dual Transformation

To bridge ES's subadditivity with the quantum solver's superadditivity requirement, we construct the **Risk Saving (RS)** characteristic function.

**Definition (RS function):**

$$\text{RS}(S) = \sum_{i \in S} \text{ES}(\{i\}) - \text{ES}(S) \tag{Def. 4.4}$$

RS measures the *diversification benefit* of coalition S: the gap between the sum of standalone risks and the portfolio risk. Since ES is subadditive, ES(S) ≤ Σ ES({i}), so RS(S) ≥ 0 for all S.

**Theorem 4.6 (Superadditivity of RS):** RS is superadditive, i.e., for disjoint S, T ⊆ N:

$$\text{RS}(S \cup T) \geq \text{RS}(S) + \text{RS}(T)$$

*Proof sketch:* Expanding both sides, the inequality reduces to ES(S) + ES(T) ≥ ES(S ∪ T), which holds by subadditivity of ES. □

The RS game is therefore compatible with the quantum Shapley solver.

> **Implementation:** `RiskSavingValueFunction.__getitem__(bitmask)` returns RS(S) for the coalition encoded by `bitmask`. Individual ES values ES({i}) are pre-computed in `__init__` and cached in `self.individual_es`.

---

## 5. Recovery Theorem

Solving the RS game yields Shapley values Φ_i^RS. We recover the original SRC via:

**Theorem 4.7:**

$$\text{SRC}_i = \text{ES}(\{i\}) - \Phi_i^{\text{RS}} \tag{Eq. 4.41}$$

*Proof:* Expanding Φ_i^RS using the Shapley formula and substituting RS(S) = Σ ES({j}) − ES(S):

$$\Phi_i^{\text{RS}} = \sum_{S \subseteq N \setminus \{i\}} \gamma(n,|S|) \left[\text{RS}(S \cup \{i\}) - \text{RS}(S)\right] = \text{ES}(\{i\}) - \Phi_i^{\text{ES}} = \text{ES}(\{i\}) - \text{SRC}_i$$

Rearranging gives the result. □

The recovery step costs O(n) — one subtraction per asset — so the overhead over the quantum Shapley computation is negligible.

> **Implementation:** `RiskAttributor(mode="rs").attribute()` computes `SRC_i = self.vfunc.individual_es[asset] - phi_list[i]` for each asset.

---

## 6. Mode Comparison

| | `mode="es"` | `mode="rs"` |
|---|---|---|
| Characteristic function | V(S) = ES(S) | RS(S) = Σ ES({i}) − ES(S) |
| Property | Subadditive | Superadditive |
| Quantum-compatible | No | **Yes** |
| Recovery step | None (SRC = Φ_i^ES) | SRC = ES({i}) − Φ_i^RS |
| Result | Identical (mathematically equivalent) | Identical |
