# Design Doc: Reduce Empty Vertical Space in Quantum Comparison Grid

## Goal
Improve readability of `docs/assets/risk/quantum_comparison_grid.png` by removing excessive upper empty space in the `n=3/4/5` panels.

## Approved Decisions
- Keep a single 2x2 composite figure.
- Keep current visual style: serif/STIX typography, solid lines, marker+color method encoding, and global legend.
- Switch y-axis scaling from global shared range to per-panel local range.
- Keep x-axis labels only on bottom row and y-axis labels only on left column.
- Keep output file path unchanged: `docs/assets/risk/quantum_comparison_grid.png`.

## Architecture and Data Flow
1. Continue loading `quantum_comparison_n3.csv` to `quantum_comparison_n6.csv` into `frames_by_n`.
2. Keep global method mapping (`method -> color`, `method -> marker`) for consistent semantics.
3. For each subplot, compute local finite `rel_error` values and derive local `y_min/y_max` with small padding.
4. Apply per-panel `set_ylim(local_min, local_max)` instead of a global y-limit.
5. Keep global legend as figure-level legend and preserve layout margins.

## Scope and Constraints
- Modify only `_plot_quantum_comparison` in `scripts/risk/plot_from_artifacts.py`.
- No changes to input CSV format, file naming conventions, or other plotting functions.

## Verification Plan
- Run: `uv run python scripts/risk/plot_from_artifacts.py`.
- Confirm `docs/assets/risk/quantum_comparison_grid.png` is generated successfully.
- Visual check: `n=3/4/5` panels should have improved vertical occupancy without large unused upper space.
