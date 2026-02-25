# Design Doc: Consolidate Quantum Comparison into 2x2 Subplots

## Goal
Replace four separate quantum comparison figures (`n=3/4/5/6`) with one publication-ready composite figure containing four subplots.

## Approved Decisions
- Layout uses a fixed 2x2 grid with subplot order: `n=3` (top-left), `n=4` (top-right), `n=5` (bottom-left), `n=6` (bottom-right).
- All subplots use a unified linear y-axis range for cross-panel comparability.
- Lines remain solid; methods are distinguished using color + marker only.
- Keep current serif scientific style (`Times New Roman` + `stix`, major/minor grids, hidden top/right spines).
- Output only one figure file: `docs/assets/risk/quantum_comparison_grid.png`.

## Architecture and Data Flow
1. Load all four CSV files first and build a shared method set.
2. Build global style mappings (`method -> color`, `method -> marker`) once.
3. Compute global y-limits from all panels and apply identical limits to each subplot.
4. Render each panel on its assigned axis with method-consistent visual encoding.
5. Save a single high-resolution figure (`dpi=300`, `bbox_inches="tight"`).

## Scope and Constraints
- Modify only `_plot_quantum_comparison` in `scripts/risk/plot_from_artifacts.py`.
- Do not change artifact inputs or naming for CSV files.
- Other plotting functions remain untouched.

## Verification Plan
- Run: `uv run python scripts/risk/plot_from_artifacts.py`
- Confirm output exists: `docs/assets/risk/quantum_comparison_grid.png`
- Confirm no per-n quantum comparison PNG files are written by this function.
