# Optimization of `_plot_quantum_comparison` Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Modify the `_plot_quantum_comparison` function to generate academic publication-quality plots matching the style of `experiments/main.pdf`.

**Architecture:** We will replace the basic Seaborn theme with a highly customized Matplotlib configuration (incorporating Times New Roman and STIX fonts, grids, tight layout, and high DPI). We will also assign unique markers and linestyles to different quantum methods for clear distinction.

**Tech Stack:** Python, Matplotlib, Seaborn, Pandas.

---

### Task 1: Update the plotting logic and styling

**Files:**
- Modify: `scripts/risk/plot_from_artifacts.py`

**Step 1: Update `_plot_quantum_comparison` function**

Modify the function to include the new style, markers, and linestyles.

```python
def _plot_quantum_comparison(snapshot_dir: str, figure_dir: str) -> None:
    asset_sizes = [3, 4, 5, 6]
    for n in asset_sizes:
        df = pd.read_csv(f"{snapshot_dir}/quantum_comparison_n{n}.csv")
        methods = df["method"].unique().tolist()
        
        # Use husl palette but map to specific markers and linestyles
        palette = dict(zip(methods, sns.color_palette("husl", len(methods))))
        markers_list = ["o", "s", "^", "D", "v", "<", ">", "p"]
        linestyles_list = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
        
        markers = {m: markers_list[i % len(markers_list)] for i, m in enumerate(methods)}
        linestyles = {m: linestyles_list[i % len(linestyles_list)] for i, m in enumerate(methods)}

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
        fig, ax = plt.subplots(figsize=(8, 5))

        for label in methods:
            subset = df[df["method"] == label]
            ax.plot(
                subset["n_l"],
                subset["rel_error"],
                marker=markers[label],
                linestyle=linestyles[label],
                label=label,
                color=palette[label],
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel(r"Interval Register Qubits ($n_l$)", fontsize=12)
        ax.set_ylabel("Mean Relative Error", fontsize=12)
        ax.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.9, edgecolor="#cccccc")
        ax.grid(True, which="major", linestyle="--", alpha=0.4)
        ax.grid(True, which="minor", linestyle=":", alpha=0.2)

        plt.tight_layout()
        out_path = f"{figure_dir}/quantum_comparison_n{n}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
```

**Step 2: Commit**

```bash
git add scripts/risk/plot_from_artifacts.py
git commit -m "style(risk): improve quantum comparison plot styling to publication quality"
```
