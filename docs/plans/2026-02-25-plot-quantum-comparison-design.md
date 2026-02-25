# Design Doc: Optimization of `_plot_quantum_comparison`

## Goal
To improve the visualization of the `_plot_quantum_comparison` function in `scripts/risk/plot_from_artifacts.py` so that it matches academic publication standards (specifically mimicking the aesthetic style of `experiments/main.pdf` and other high-quality plots in the project like `_plot_equal_error`).

## Decisions
- **Approach:** Direct Matplotlib native plotting with explicit styling mapping, rather than a purely automated Seaborn default output.
- **Font:** `Times New Roman` for normal text and `stix` for mathematical formulas to give it a LaTeX feel.
- **Styling:**
  - Utilize distinct `linestyle` (solid, dashed, dotted, dashdot) and `marker` ('o', 's', '^', 'D', 'v') for each line (i.e. for each `method`).
  - High resolution (`dpi=300`) and tight bounding box (`bbox_inches="tight"`) for exporting images.
  - Adding major and minor background grids (`axes.grid: True`).
- **Data Scaling:** Maintaining original axes scaling for now but with more professional tick labels and legends.

## Implementation Details
1. In `scripts/risk/plot_from_artifacts.py`, replace the `sns.set_theme` call inside `_plot_quantum_comparison` to match the advanced configuration:
   ```python
   sns.set_theme(
       style="whitegrid",
       context="paper",
       font_scale=1.5,
       rc={
           "font.family": "serif",
           "font.serif": ["Times New Roman"],
           "mathtext.fontset": "stix",
           "axes.grid": True,
           "grid.linestyle": "--",
           "grid.alpha": 0.4,
       },
   )
   ```
2. Define a set of markers and linestyles to iterate through, mapping them securely to each unique method retrieved from the CSV.
3. Update `ax.plot` to consume `color`, `marker`, and `linestyle`.
4. Modify the output `plt.savefig()` to include `dpi=300, bbox_inches="tight"`.