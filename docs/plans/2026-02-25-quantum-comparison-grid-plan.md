# Quantum Comparison 2x2 Grid Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 `_plot_quantum_comparison` 从输出 4 张独立图改为输出 1 张包含 4 个子图的总图，并保持统一线性坐标可比性。

**Architecture:** 保留现有数据来源（4 个 `quantum_comparison_n*.csv`）和样式体系，先全量读取数据建立统一的 `method -> color/marker` 映射，再在单个 Figure 中按固定位置绘制 2x2 子图。所有子图使用同一组 y 轴范围，保证跨子图横向比较有效。输出从多文件改为单文件 `quantum_comparison_grid.png`。

**Tech Stack:** Python 3.11, Matplotlib, Seaborn, Pandas, pytest.

---

### Task 1: Add regression test for single-grid output contract

**Files:**
- Create: `tests/double_quant/application/test_risk_plot_from_artifacts.py`
- Modify: `scripts/risk/plot_from_artifacts.py`

**Step 1: Write the failing test**

```python
from pathlib import Path
import pandas as pd

from scripts.risk.plot_from_artifacts import _plot_quantum_comparison


def _write_snapshot(path: Path, n: int) -> None:
    df = pd.DataFrame(
        {
            "n_l": [2, 3, 4],
            "method": ["qae_fae", "qae_fae", "qae_fae"],
            "rel_error": [0.4, 0.2, 0.1],
        }
    )
    df.to_csv(path / f"quantum_comparison_n{n}.csv", index=False)


def test_quantum_comparison_outputs_single_grid_figure(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    figure_dir = tmp_path / "figure"
    snapshot_dir.mkdir()
    figure_dir.mkdir()

    for n in (3, 4, 5, 6):
        _write_snapshot(snapshot_dir, n)

    _plot_quantum_comparison(str(snapshot_dir), str(figure_dir))

    assert (figure_dir / "quantum_comparison_grid.png").exists()
    assert not (figure_dir / "quantum_comparison_n3.png").exists()
    assert not (figure_dir / "quantum_comparison_n4.png").exists()
    assert not (figure_dir / "quantum_comparison_n5.png").exists()
    assert not (figure_dir / "quantum_comparison_n6.png").exists()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/double_quant/application/test_risk_plot_from_artifacts.py::test_quantum_comparison_outputs_single_grid_figure -v`
Expected: FAIL because current implementation still writes `quantum_comparison_n{n}.png`.

**Step 3: Commit (test-only, red phase)**

```bash
git add tests/double_quant/application/test_risk_plot_from_artifacts.py
git commit -m "test(risk): add failing contract test for quantum comparison grid output"
```

### Task 2: Refactor plotting to render one 2x2 figure

**Files:**
- Modify: `scripts/risk/plot_from_artifacts.py:145-223`
- Test: `tests/double_quant/application/test_risk_plot_from_artifacts.py`

**Step 1: Implement minimal code to satisfy test**

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
panel_order = [3, 4, 5, 6]

# precompute global y limits
all_values = []
for n in panel_order:
    frame = frames_by_n[n]
    all_values.extend(np.asarray(frame["rel_error"]).tolist())
y_min, y_max = min(all_values), max(all_values)

for ax, n in zip(axes.flatten(), panel_order, strict=False):
    df = frames_by_n[n]
    ...  # draw methods with shared color/marker mapping
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"n = {n}")

fig.savefig(f"{figure_dir}/quantum_comparison_grid.png", dpi=300, bbox_inches="tight")
```

Implementation requirements:
- Keep line style solid (`linestyle="-"`).
- Keep method distinction by marker and color.
- Keep serif/STIX/theme/grid style.
- Left column keeps y-label; bottom row keeps x-label.

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/double_quant/application/test_risk_plot_from_artifacts.py::test_quantum_comparison_outputs_single_grid_figure -v`
Expected: PASS.

**Step 3: Run script-level verification**

Run: `uv run python scripts/risk/plot_from_artifacts.py`
Expected: console prints saved path for `quantum_comparison_grid.png`, no `quantum_comparison_n*.png` from this function.

**Step 4: Commit implementation**

```bash
git add scripts/risk/plot_from_artifacts.py tests/double_quant/application/test_risk_plot_from_artifacts.py
git commit -m "feat(risk): merge quantum comparison plots into 2x2 grid figure"
```
