# Shapley Value Computation

## 1. Shapley Value

Given an n-player cooperative game (N, v), the **Shapley value** Φ_i(v) of player i is the unique allocation satisfying efficiency, symmetry, dummy, and additivity axioms:

$$\Phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \gamma(n, |S|)\,[v(S \cup \{i\}) - v(S)] \tag{Eq. 4.4}$$

where the **Shapley weight** is:

$$\gamma(n, s) = \frac{s!\,(n - s - 1)!}{n!} \tag{Eq. 4.5}$$

γ(n, s) is the probability that coalition S appears as the set of predecessors of player i in a uniformly random permutation of N.

### Classical Complexity

Exact classical computation requires evaluating v on all 2^n subsets and summing n × 2^(n−1) weighted marginal contributions — complexity O(n · 2^n). For portfolios with n ≥ 20 assets this becomes prohibitive.

> **Classical implementations:** `BinaryEnumerationCalculator` (subset enumeration) and `PermutationEnumerationCalculator` (permutation enumeration) in `double_quant.solver.shapley`.

---

## 2. Quantum Shapley Algorithm

The quantum algorithm (following Burge et al.) estimates Φ_i for a single target player i in one circuit evaluation, achieving a quadratic speedup in the number of function queries via quantum amplitude estimation (QAE).

### Superadditivity Requirement

The algorithm encodes the value function as rotation angles in a quantum circuit. The encoding is valid only when v is **superadditive**:

$$v(S \cup T) \geq v(S) + v(T) \quad \forall\, S \cap T = \emptyset$$

For risk attribution with ES as the game, use the **Risk Saving transformation** (see [docs/application/risk.md](../application/risk.md)) to obtain a superadditive characteristic function before passing it to `QuantumCalculator`.

---

## 3. Continuous Integral Reformulation

The key observation is that the Shapley weight γ(n, s) can be written as an integral over [0, 1]:

**Definition 4.1 (β-weight):**

$$\beta(n, s) = \int_0^1 \binom{n-1}{s} t^s (1-t)^{n-1-s}\, dt$$

**Theorem 4.1:** γ(n, s) = β(n, s) for all n ≥ 1 and 0 ≤ s ≤ n − 1.

This lets us write the Shapley value as a continuous integral:

$$\Phi_i(v) = \int_0^1 \sum_{S \subseteq N \setminus \{i\},\, |S|=k(t)} [v(S \cup \{i\}) - v(S)]\, dt$$

where the inner sum averages marginal contributions over all coalitions of the size determined by t. This integral form is what the quantum circuit approximates.

---

## 4. Interval Discretization

To approximate the integral with a quantum circuit of n_l internal qubits, the [0, 1] interval is divided into 2^(n_l) sub-intervals with **sin²-spaced** sample points.

**Definition 4.2 (sin² sample points):**

$$t_j = \sin^2\!\left(\frac{\pi}{2} \cdot \frac{2j+1}{2^{n_l+1}}\right), \quad j = 0, 1, \ldots, 2^{n_l}-1$$

**Definition 4.3 (discretized β-weight):**

$$\tilde{\beta}(n, s, n_l) = \frac{1}{2^{n_l}} \sum_{j=0}^{2^{n_l}-1} \binom{n-1}{s} t_j^s (1 - t_j)^{n-1-s}$$

**Theorem 4.5 (Convergence):** As n_l → ∞, the discretized approximation recovers the exact Shapley value:

$$\lim_{n_l \to \infty} \tilde{\Phi}_i = \Phi_i(v)$$

In practice n_l = O(log n) internal qubits suffice for acceptable precision.

> **Implementation:** `internal_qubits_num` in `QuantumCalculator.__init__` controls n_l. Default: `int(log2(num_players) * internal_multiplier)` with `internal_multiplier=2`.

---

## 5. Quantum Circuit Structure

The circuit uses three quantum registers:

| Register | Size | Role |
|---|---|---|
| Q_l (internal) | n_l qubits | Encodes the 2^(n_l) discretization points t_j via `IntervalLoader` |
| Q_p (player) | n − 1 qubits | Encodes coalition membership for the n − 1 players other than target i via `VertexRotator` |
| Q_a (output) | 1 qubit | Accumulates the weighted marginal contribution; measured at the end |

### IntervalLoader

Prepares the superposition over discretization points:

$$|\psi_l\rangle = \frac{1}{\sqrt{2^{n_l}}} \sum_{j=0}^{2^{n_l}-1} |j\rangle$$

using a uniform `StatePreparation` gate, with weights chosen so that measuring Q_l in basis state |j⟩ gives sample point t_j.

### VertexRotator

Conditioned on Q_l register state |j⟩, applies controlled-RY rotations to Q_p so that the probability of each Q_p basis state |S⟩ (encoding coalition S) equals the Bernoulli probability of that coalition size at point t_j:

$$P(|S\rangle \mid t_j) = t_j^{|S|}(1-t_j)^{n-1-|S|}$$

### ValueLoader (U_W gate)

Encodes the marginal contributions into Q_a via a uniformly controlled RY rotation:

$$U_W: |S\rangle|0\rangle \mapsto |S\rangle \left(\cos\theta_S|0\rangle + \sin\theta_S|1\rangle\right)$$

where the rotation angle is:

$$\theta_S = \arcsin\!\sqrt{\frac{v(S \cup \{i\}) - v(S)}{W_{\max}}} \tag{Eq. 4.26}$$

and W_max = max_S [v(S ∪ {i}) − v(S)] normalizes contributions to [0, 1].

---

## 6. Measurement and Extraction

After the full circuit evolution, the probability of measuring Q_a = |1⟩ is:

$$P(|1\rangle) = \frac{\tilde{\Phi}_i}{W_{\max}} \tag{Eq. 4.35}$$

The Shapley value estimate is therefore:

$$\tilde{\Phi}_i = W_{\max} \cdot P(|1\rangle)$$

In the current implementation this probability is read directly from a statevector simulation (Qiskit Aer). In a hardware deployment, **Quantum Amplitude Estimation (QAE)** would extract P(|1⟩) with O(1/ε) circuit repetitions, achieving a quadratic speedup over classical Monte Carlo's O(1/ε²).

> **Implementation:** `QuantumCalculator._calculate_one(target_player)` in `double_quant.solver.shapley` builds and simulates this circuit for each target player.
