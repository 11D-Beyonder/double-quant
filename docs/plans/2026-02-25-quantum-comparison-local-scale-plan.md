# Quantum Comparison Local Y-Scale Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 优化 `quantum_comparison_grid.png`，让每个子图使用局部线性 y 轴范围，减少 `n=3/4/5` 面板上半部分空白。

**Architecture:** 保留现有 2x2 总图、全局图例、统一方法视觉编码（color+marker）和论文样式，只调整 y 轴策略：从全局共享范围改为每个子图按自身数据计算局部范围。为防止边界异常，继续过滤有限值并保留小幅 padding。输出路径与文件名不变。

**Tech Stack:** Python 3.11, Matplotlib, Seaborn, Pandas, pytest.

---

### Task 1: Add regression test for panel-local y-limits

**Files:**
- Modify: `tests/double_quant/application/test_risk_plot_from_artifacts.py`
- Test: `tests/double_quant/application/test_risk_plot_from_artifacts.py`

**Step 1: Write the failing test**

在现有测试文件新增一个测试，构造 n=3/4/5 的低范围数据和 n=6 的高范围数据，并通过 `monkeypatch` 捕获 `_plot_quantum_comparison` 内部创建的 `fig, axes`，断言四个子图的 y-limit 不完全相同（至少 `n=6` 与 `n=3` 有明显差异）。

```python
def test_quantum_comparison_uses_local_panel_y_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}
    original_subplots = plt.subplots

    def _capture_subplots(*args, **kwargs):
        fig, axes = original_subplots(*args, **kwargs)
        captured["fig"] = fig
        captured["axes"] = axes
        return fig, axes

    monkeypatch.setattr(plt, "subplots", _capture_subplots)
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    # prepare snapshots with panel-dependent ranges ...
    _plot_quantum_comparison(str(snapshot_dir), str(figure_dir))

    axes = captured["axes"].flatten()
    y_limits = [ax.get_ylim() for ax in axes]
    assert len(set(y_limits)) > 1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/double_quant/application/test_risk_plot_from_artifacts.py::test_quantum_comparison_uses_local_panel_y_limits -v`
Expected: FAIL（当前实现仍统一 y 轴，ylim 相同）。

**Step 3: Commit (optional, if user requests commit flow)**

```bash
git add tests/double_quant/application/test_risk_plot_from_artifacts.py
git commit -m "test(risk): add failing test for local y-scale panel behavior"
```

### Task 2: Switch grid plotting from global y-limits to local per-panel limits

**Files:**
- Modify: `scripts/risk/plot_from_artifacts.py:178-224`
- Test: `tests/double_quant/application/test_risk_plot_from_artifacts.py`

**Step 1: Write minimal implementation**

在每个子图循环内计算本地范围：
- 删除全局 `all_rel_error_values` + `y_limits` 的统一计算逻辑。
- 在每个 panel 内按当前 `df["rel_error"]` 过滤 finite 值并计算 `local_min/local_max`。
- 使用现有 padding 规则（相等值时最小 padding）后调用 `ax.set_ylim(local_min - pad, local_max + pad)`。

```python
panel_values = np.asarray(df["rel_error"], dtype=float)
panel_values = panel_values[np.isfinite(panel_values)]
if panel_values.size == 0:
    raise ValueError(f"No finite rel_error values for n={n}")

local_min = float(np.min(panel_values))
local_max = float(np.max(panel_values))
if np.isclose(local_min, local_max):
    local_pad = max(abs(local_min) * 0.05, 1e-6)
else:
    local_pad = 0.05 * (local_max - local_min)
ax.set_ylim(local_min - local_pad, local_max + local_pad)
```

同时将 `plt.subplots(..., sharey=True)` 改为 `sharey=False`。

**Step 2: Run tests to verify pass**

Run: `uv run pytest tests/double_quant/application/test_risk_plot_from_artifacts.py -v`
Expected: PASS（包含旧的单输出契约测试与新的 local-scale 测试）。

**Step 3: Run rendering verification**

Run: `uv run python scripts/risk/plot_from_artifacts.py`
Expected: 成功生成 `docs/assets/risk/quantum_comparison_grid.png`，且 `n=3/4/5` 面板顶部空白明显减少。

**Step 4: Commit (optional, if user requests commit flow)**

```bash
git add scripts/risk/plot_from_artifacts.py tests/double_quant/application/test_risk_plot_from_artifacts.py docs/assets/risk/quantum_comparison_grid.png
git commit -m "feat(risk): use local y-scale for quantum comparison subplots"
```
