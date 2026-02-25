# Quantum Comparison Inset Zoom Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在保持 2x2 主图统一线性 y 轴可比性的前提下，为每个子图增加 inset 局部放大，解决 `n=3/4/5` 面板上半区过空的问题。

**Architecture:** 保留现有主图数据流（统一方法映射、统一 y 范围、单输出文件），在每个主轴上新增一个右上角 inset 轴，复用同一批曲线数据绘制局部细节。inset 的 y 范围按子图内数据自动计算（默认 0 到 P95 并加 padding），x 轴保持完整范围，避免趋势断裂。图例继续使用全局 legend，不在 inset 重复。

**Tech Stack:** Python 3.11, Matplotlib, Seaborn, NumPy, Pandas, pytest.

---

### Task 1: Add failing regression test for inset creation behavior

**Files:**
- Modify: `tests/double_quant/application/test_risk_plot_from_artifacts.py`
- Test: `tests/double_quant/application/test_risk_plot_from_artifacts.py`

**Step 1: Write the failing test**

Add a focused test that verifies inset axes are created once per panel (4 times) when `_plot_quantum_comparison` runs.

```python
from matplotlib.axes import Axes


def test_quantum_comparison_creates_inset_per_panel(tmp_path: Path, monkeypatch) -> None:
    snapshot_dir = tmp_path / "snapshot"
    figure_dir = tmp_path / "figure"
    snapshot_dir.mkdir()
    figure_dir.mkdir()

    for n in (3, 4, 5, 6):
        _write_snapshot(snapshot_dir, n)

    call_count = {"value": 0}
    original_inset_axes = Axes.inset_axes

    def _counting_inset_axes(self, *args, **kwargs):
        call_count["value"] += 1
        return original_inset_axes(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "inset_axes", _counting_inset_axes)

    _plot_quantum_comparison(str(snapshot_dir), str(figure_dir))

    assert call_count["value"] == 4
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/double_quant/application/test_risk_plot_from_artifacts.py::test_quantum_comparison_creates_inset_per_panel -v`
Expected: FAIL because current implementation has no inset creation.

**Step 3: Commit (red phase)**

```bash
git add tests/double_quant/application/test_risk_plot_from_artifacts.py
git commit -m "test(risk): add failing test for quantum comparison inset axes"
```

### Task 2: Implement inset zoom in `_plot_quantum_comparison`

**Files:**
- Modify: `scripts/risk/plot_from_artifacts.py:146-271`
- Test: `tests/double_quant/application/test_risk_plot_from_artifacts.py`

**Step 1: Implement minimal code to satisfy inset behavior**

Within each subplot loop:
- Create inset axis in top-right via `ax.inset_axes([0.56, 0.53, 0.4, 0.42])`.
- Plot the same method curves on inset using existing `plot_kwargs`.
- Compute local inset y-range from panel values:
  - `local_values = np.asarray(df["rel_error"], dtype=float)`
  - filter finite values
  - `upper = np.percentile(local_values, 95)`
  - apply small padding and clamp lower to 0.0
- Set inset x-range to panel x min/max and y-range to computed local window.
- Keep inset ticks sparse and small, no inset legend.

Reference implementation shape:

```python
inset_ax = ax.inset_axes([0.56, 0.53, 0.4, 0.42])
for label in methods_in_plot:
    inset_ax.plot(x_vals[order], y_vals[order], **plot_kwargs)

finite_local = local_values[np.isfinite(local_values)]
upper = float(np.percentile(finite_local, 95))
pad = max(upper * 0.08, 1e-6)
inset_ax.set_ylim(0.0, upper + pad)
inset_ax.set_xlim(float(np.min(x_all)), float(np.max(x_all)))
inset_ax.grid(True, which="major", linestyle=":", alpha=0.25)
```

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/double_quant/application/test_risk_plot_from_artifacts.py::test_quantum_comparison_creates_inset_per_panel -v`
Expected: PASS.

**Step 3: Run existing grid contract test**

Run: `uv run pytest tests/double_quant/application/test_risk_plot_from_artifacts.py::test_quantum_comparison_outputs_single_grid_figure -v`
Expected: PASS (still only outputs `quantum_comparison_grid.png`).

**Step 4: Run script-level verification**

Run: `uv run python scripts/risk/plot_from_artifacts.py`
Expected:
- output file: `docs/assets/risk/quantum_comparison_grid.png`
- no per-n `quantum_comparison_n*.png` produced by this function.

**Step 5: Commit implementation**

```bash
git add scripts/risk/plot_from_artifacts.py tests/double_quant/application/test_risk_plot_from_artifacts.py docs/assets/risk/quantum_comparison_grid.png
git commit -m "feat(risk): add inset zoom panels to quantum comparison grid"
```
