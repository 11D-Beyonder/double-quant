# Risk Equal-Error Four-Method Comparison Design

## Background

Current risk experiment artifacts provide:

- Quantum method comparison across `statevector`, `shots`, `IQAE`, `MLQAE`, `FAE`
- Equal-error oracle-call summary for only `Classical MC` and `ML-QAE`

Observed issue: IQAE appears underperforming in `quantum_comparison_n*.png`, largely due to a coarse IQAE setting (`epsilon=0.05`) and non-equal-budget comparisons.

## Goals

1. Improve IQAE setting in stage-1 quantum comparison to a stricter and more stable target.
2. Extend equal-error analysis to four methods:
   - Classical MC
   - IQAE
   - ML-QAE
   - FAE
3. Keep error metric consistent with existing pipeline: mean relative error (MRE).
4. Reuse existing artifact and plotting pipeline with minimal structural disruption.

## Non-Goals

- No changes to the mathematical definition of risk attribution.
- No replacement of MRE with MAE in this iteration.
- No new artifact file family unless strictly necessary.

## Design Decisions

### 1) IQAE parameter update in quantum comparison stage

In `_generate_quantum_comparison_snapshots`:

- Change IQAE options from `QAEOptions(epsilon=0.05, alpha=0.05)`
- To `QAEOptions(epsilon=0.01, alpha=0.01)`

Rationale:

- Reduce frequent low-query or degenerate behavior under too-loose epsilon.
- Improve IQAE accuracy so stage-1 comparison is more representative.

### 2) Extend equal-error snapshot from 2 methods to 4 methods

In `_generate_equal_error_snapshot`:

- Keep existing per-round workflow:
  - sample portfolio
  - compute exact reference via `BinaryEnumerationCalculator`
  - generate `(oracle_calls, error)` points per method
  - derive minimum calls for each target epsilon
- Expand methods from `{Classical MC, ML-QAE}` to:
  - `Classical MC`
  - `IQAE`
  - `ML-QAE`
  - `FAE`

### 3) Parameter grids for each method

- Classical MC:
  - keep existing `classical_samples`
- IQAE:
  - scan epsilon grid, fixed alpha
  - recommended grid: `[0.05, 0.03, 0.02, 0.01, 0.007, 0.005]`
  - fixed `alpha=0.01`
- ML-QAE:
  - keep `num_eval_qubits` grid, optionally include a wider range `[2, 3, 4, 5, 6]`
- FAE:
  - fixed `delta=0.05`
  - scan `maxiter` grid `[3, 4, 5, 6, 7]`

Rationale:

- Generate a richer and monotonic-like cost-error frontier for each method.
- Reduce over-reliance on fallback extrapolation.

### 4) Error and aggregation semantics remain unchanged

- Error metric remains MRE via `_mean_relative_error`.
- Reachability logic remains:
  - direct discrete hit from sampled points, else
  - log-log fallback from `_fallback_calls_from_loglog_fit`.
- Summary schema remains unchanged:
  - `method, epsilon, mean_calls, std_calls, reachable_ratio, source_type`
- Output file remains unchanged:
  - `docs/assets/risk/data/equal_error_oracle_calls_summary.csv`

Rationale:

- Preserve downstream compatibility and reduce migration effort.

### 5) Plotting update for four-method equal-error chart

In `_plot_equal_error`:

- Expand plotted method list to four methods.
- Extend fixed palette mapping to include all four entries.
- Keep axis semantics and scales unchanged:
  - `epsilon` on log-x
  - oracle calls on log-y
- Keep output filename unchanged:
  - `docs/assets/risk/equal_error_oracle_calls_fixed_grid_fallback.png`

Rationale:

- Single chart directly answers the equal-error oracle-call comparison request.
- No need to update artifact lookup conventions or external references.

## Data Flow

1. `scripts/risk/generate_artifacts.py` computes snapshots.
2. Equal-error snapshot now includes 4-method summary rows in the same CSV.
3. `scripts/risk/plot_from_artifacts.py` reads the same CSV and renders 4 curves.
4. Manifest captures updated parameter configuration for reproducibility.

## Failure Handling

- If a method/parameter combination raises an exception, skip that point and continue.
- If a target epsilon is unreachable and fallback is invalid, keep `oracle_calls` as missing (`none` source type).
- Never fail the full snapshot generation because a subset of quantum points is invalid.

## Test and Verification Strategy

- Regenerate artifacts with `--force`.
- Check summary CSV contains all four method labels.
- Validate there are rows for each `(method, epsilon)` pair.
- Render figures and confirm 4 lines appear in equal-error chart legend.
- Spot-check that IQAE in quantum-comparison snapshots reflects updated options.

## Risks

- Runtime increase from extra quantum grid evaluations.
- Some epsilon levels may still be sparse for selected methods in certain rounds.
- MRE can remain sensitive when exact values are near zero (accepted in this iteration by design).

## Rollout Plan

1. Update artifact generation parameters and method loops.
2. Update equal-error plotting for 4 methods.
3. Regenerate snapshots and figures.
4. Review output quality and adjust grids only if required.
