# Design Doc: Equal-Error I-QAE Sampling Reduction Reporting

## Goal
Add console reporting in `_plot_equal_error` to quantify how much I-QAE reduces sampling cost versus Classical MC under the same target relative error (`epsilon`).

## Approved Decisions
- Output style is **full per-epsilon table** plus **max-reduction summary**.
- Use **direct epsilon matching** between `Classical MC` and `I-QAE` (no interpolation).
- Keep scope limited to `scripts/risk/plot_from_artifacts.py` and only `_plot_equal_error`.
- Preserve current plotting behavior and output image generation.

## Approach Options and Trade-offs
1. **Direct matching (selected):** Join rows by exact `epsilon`.
   - Pros: deterministic, transparent, no model assumptions.
   - Cons: cannot compare rows when epsilon grids differ.
2. Interpolation alignment: Estimate one curve on the other's epsilon grid.
   - Pros: more complete comparison when grids differ.
   - Cons: introduces approximation bias and interpretability risk.
3. Persist comparison CSV: write per-epsilon reductions to artifact file.
   - Pros: reusable downstream.
   - Cons: increases output surface and maintenance burden.

## Architecture and Data Flow
Within `_plot_equal_error`:
1. Load `equal_error_oracle_calls_summary.csv` as today.
2. Normalize method names (`IQAE` -> `I-QAE`, `FAE` -> `F-QAE`) as today.
3. Extract valid `epsilon` + `mean_calls` for:
   - `Classical MC`
   - `I-QAE`
4. Inner-join on `epsilon` to obtain comparable rows.
5. Compute reduction percentage per row:

   `reduction_pct = (mc_calls - iqae_calls) / mc_calls * 100`

6. Print per-epsilon comparison rows in ascending `epsilon`.
7. Print the max reduction row with epsilon and both call counts.
8. Continue with existing plotting logic unchanged.

## Error Handling
- If either method data is missing: print warning and skip comparison block.
- If matched rows are empty: print warning and skip comparison block.
- If `Classical MC mean_calls <= 0`: skip those rows to avoid invalid ratios.
- Filter `NaN` / non-finite values before calculations.
- Never fail plotting due to reporting-only issues.

## Output Format
- Header line for the comparison block.
- Per-epsilon lines include:
  - `epsilon`
  - `Classical MC mean_calls`
  - `I-QAE mean_calls`
  - `reduction_pct`
- Final line reports maximum reduction and corresponding epsilon.

## Scope and Non-Goals
- In scope: console reporting inside `_plot_equal_error`.
- Out of scope:
  - interpolation-based comparison,
  - writing new CSV outputs,
  - new CLI flags or behavior changes in other plotting functions.

## Verification Plan
- Run:

  `uv run scripts/risk/plot_from_artifacts.py`

- Expect:
  - figure still saved to `docs/assets/risk/equal_error_oracle_calls_fixed_grid_fallback.png`,
  - console shows per-epsilon reduction rows,
  - console shows one max-reduction summary line.
