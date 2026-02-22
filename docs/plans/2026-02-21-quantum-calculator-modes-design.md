# QuantumCalculator Multi-Mode Design

**Date:** 2026-02-21
**Branch:** feat/joe/shapley
**Status:** Approved

## Context

The current `QuantumCalculator` extracts Shapley values by directly reading probabilities from a `Statevector` simulation — an exact, noiseless method that does not reflect the query complexity of a real quantum device. To support the complexity and acceleration advantage analysis described in `tests/double_quant/application/EXPERIMENT.md`, three additional extraction modes are needed:

- **shots**: Simulates shot-based quantum measurement via Qiskit primitives
- **qae_canonical / qae_iqae / qae_mlqae**: Three variants of Quantum Amplitude Estimation from `qiskit-algorithms`

All modes are unified under a single `QuantumCalculator` class with a `mode` parameter.

## New Dependency

```
uv add qiskit-algorithms
```

The QAE implementations (`AmplitudeEstimation`, `IterativeAmplitudeEstimation`, `MaximumLikelihoodAmplitudeEstimation`, `EstimationProblem`) are in `qiskit-algorithms` (Qiskit Ecosystem), not the main `qiskit` package.

## Architecture

### `QAEOptions` dataclass

Collects all mode-specific parameters to avoid bloating `QuantumCalculator.__init__`:

```python
@dataclass
class QAEOptions:
    # shots mode
    shots: int = 1024

    # all QAE modes
    epsilon: float = 0.01   # target precision (half confidence interval width)
    alpha: float = 0.05     # confidence level (IQAE / MLQAE)

    # canonical QAE only
    num_eval_qubits: int = 3  # QPE precision; oracle calls ~ 2^num_eval_qubits
```

### `QuantumCalculator` changes

```python
Mode = Literal["statevector", "shots", "qae_canonical", "qae_iqae", "qae_mlqae"]

class QuantumCalculator(ShapleyCalculator):
    def __init__(
        self,
        num_players: int,
        value_dict: ValueFunction | None = None,
        internal_qubits_num: int | None = None,
        internal_multiplier: float = 2,
        mode: Mode = "statevector",        # extraction mode
        options: QAEOptions | None = None, # mode-specific parameters
    ): ...
```

Oracle call counts are tracked alongside the existing `_shapley_cache`:

```python
self._oracle_call_counts: list[int | None] = [None] * num_players
```

Exposed via:

```python
def get_oracle_count(self, player_index: int) -> int | None: ...
```

`player_index` is kept per-player because IQAE is adaptive: different players may require different numbers of Grover iterations to converge to the same precision `epsilon`.

## Data Flow

Circuit construction (`_init_circuit` + `_extend_circuit`) is unchanged across all modes. Only the amplitude extraction step branches:

```
_calculate_one(player_index)
    │
    ├─ build: _init_circuit() + _extend_circuit(player_index)
    │
    └─ extract amplitude (branch on self.mode):
        │
        ├─ "statevector"
        │     Statevector(circuit).probabilities()[output_qubit == 1]
        │     oracle_count ← 1
        │
        ├─ "shots"
        │     StatevectorSampler(shots=options.shots).run(circuit + measure)
        │     count |1⟩ / total_shots
        │     oracle_count ← options.shots
        │
        ├─ "qae_canonical"
        │     AmplitudeEstimation(num_eval_qubits=options.num_eval_qubits, sampler=...)
        │         .estimate(EstimationProblem(state_preparation=circuit,
        │                                    objective_qubits=[output_qubit]))
        │     oracle_count ← result.num_oracle_queries
        │
        ├─ "qae_iqae"
        │     IterativeAmplitudeEstimation(epsilon_target=options.epsilon,
        │                                  alpha=options.alpha, sampler=...)
        │         .estimate(EstimationProblem(...))
        │     oracle_count ← result.num_oracle_queries
        │
        └─ "qae_mlqae"
              MaximumLikelihoodAmplitudeEstimation(evaluation_schedule=..., sampler=...)
                  .estimate(EstimationProblem(...))
              oracle_count ← result.num_oracle_queries
```

**`EstimationProblem` setup:**
- `state_preparation` = full circuit (no measurement)
- `objective_qubits` = `[output_qubit_index]` (last qubit in the circuit)
- `grover_operator` = not provided; qiskit-algorithms auto-derives it from `state_preparation`

## Error Handling

Minimal, at system boundaries only:

- **Invalid `mode`**: Enforced by `Literal` type hint; no runtime check needed.
- **`options=None` when mode requires it**: `_calculate_one` raises `ValueError` before accessing `options`:
  ```
  ValueError: "QAEOptions required for mode='qae_iqae'"
  ```
- **`get_oracle_count` called before computation**: Returns `None`; caller is responsible for checking.

## Testing

Extend the existing `TestDifferentSolver` class in `tests/double_quant/application/test_risk.py`:

| Test | Description |
|---|---|
| `test_shots_convergence` | 3-player portfolio; shots = [256, 1024, 4096]; verify error vs. statevector ground truth decreases with more shots |
| `test_qae_modes_basic` | Same portfolio; run all three QAE modes; verify result within `epsilon * 2` of statevector |
| `test_oracle_count_tracked` | After each mode run, `get_oracle_count` is not `None`; shots mode count equals `options.shots` |

Complexity plots (oracle count vs. epsilon across modes) are added to the existing experiment suite in `EXPERIMENT.md`.
