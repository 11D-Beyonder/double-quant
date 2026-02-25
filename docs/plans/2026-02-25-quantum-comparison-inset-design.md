# Design Doc: Inset Zoom for Quantum Comparison Grid

## Goal
Improve readability of `docs/assets/risk/quantum_comparison_grid.png` when upper space in `n=3/4/5` panels appears too empty under unified linear y-axis scaling.

## Approved Design
- Keep the current 2x2 main layout and unified linear y-axis limits across all panels.
- Add one inset zoom area to each panel (top-right corner) to reveal low-error region details.
- Keep global legend only on the main figure; do not place legends in inset axes.
- Keep solid lines with marker+color method encoding and current serif scientific theme.

## Inset Scaling Rules
- For each panel, compute inset y-range from local panel data.
- Default local zoom window: `0` to `P95(rel_error)` with small top padding.
- Keep full x-range in inset (`n_l` not cropped) so trend continuity remains visible.
- If panel values are flat or degenerate, apply a minimum epsilon padding to avoid identical y-limits.

## Scope
- Modify only `_plot_quantum_comparison` in `scripts/risk/plot_from_artifacts.py`.
- Keep artifact CSV input structure unchanged.
- Keep output filename unchanged: `docs/assets/risk/quantum_comparison_grid.png`.

## Verification
- Run: `uv run python scripts/risk/plot_from_artifacts.py`
- Confirm output file updated: `docs/assets/risk/quantum_comparison_grid.png`
- Visual acceptance criteria:
  - Main panels retain unified linear y-axis comparability.
  - Insets show clearer separation in low-error region for `n=3/4/5`.
  - Figure remains readable without legend overlap.
