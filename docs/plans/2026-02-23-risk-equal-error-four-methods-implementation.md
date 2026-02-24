# Risk Equal-Error Four-Method Comparison Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update IQAE defaults in quantum comparison and extend equal-error oracle-call benchmarking/plotting from 2 methods to 4 methods (Classical MC, IQAE, ML-QAE, FAE) under the same MRE metric.

**Architecture:** Keep the existing artifact pipeline and output file names stable. Expand method grids inside `scripts/risk/generate_artifacts.py` and reuse the same summary schema (`equal_error_oracle_calls_summary.csv`). Update plotting logic in `scripts/risk/plot_from_artifacts.py` to render 4 curves from the same summary file.

**Tech Stack:** Python 3.11, NumPy, Pandas, Qiskit (`qiskit`, `qiskit-algorithms`), Matplotlib/Seaborn, Pytest.

---

### Task 1: Tighten IQAE config in quantum comparison stage

**Files:**
- Modify: `scripts/risk/generate_artifacts.py`
- Test: `tests/double_quant/application/test_risk_artifacts.py`

**Step 1: Write the failing test**

Add a regression test that asserts the script encodes the new IQAE configuration in source text.

```python
def test_generate_artifacts_uses_tighter_iqae_in_quantum_comparison():
    content = Path("scripts/risk/generate_artifacts.py").read_text()
    assert 'QAEOptions(epsilon=0.01, alpha=0.01)' in content
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/double_quant/application/test_risk_artifacts.py::test_generate_artifacts_uses_tighter_iqae_in_quantum_comparison -v`
Expected: FAIL because script still has `epsilon=0.05, alpha=0.05`.

**Step 3: Write minimal implementation**

In `scripts/risk/generate_artifacts.py`, update the `methods` list inside `_generate_quantum_comparison_snapshots`:

```python
("qae_iqae", QAEOptions(epsilon=0.01, alpha=0.01), "qae_iqae"),
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/double_quant/application/test_risk_artifacts.py::test_generate_artifacts_uses_tighter_iqae_in_quantum_comparison -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/risk/generate_artifacts.py tests/double_quant/application/test_risk_artifacts.py
git commit -m "fix(risk): tighten iqae settings in quantum comparison"
```

### Task 2: Extend equal-error snapshot computation to 4 methods

**Files:**
- Modify: `scripts/risk/generate_artifacts.py`
- Test: `tests/double_quant/application/test_risk_artifacts.py`

**Step 1: Write the failing test**

Add a contract test that checks equal-error section contains all four method labels and method grids.

```python
def test_generate_artifacts_equal_error_mentions_four_methods():
    content = Path("scripts/risk/generate_artifacts.py").read_text()
    assert '"Classical MC"' in content
    assert '"IQAE"' in content
    assert '"ML-QAE"' in content
    assert '"FAE"' in content
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/double_quant/application/test_risk_artifacts.py::test_generate_artifacts_equal_error_mentions_four_methods -v`
Expected: FAIL because `IQAE`/`FAE` are absent in equal-error method list.

**Step 3: Write minimal implementation**

Refactor `_generate_equal_error_snapshot` to build points for all methods:

```python
points: dict[str, list[tuple[int, float]]] = {
    "Classical MC": [],
    "IQAE": [],
    "ML-QAE": [],
    "FAE": [],
}
```

Add method-specific grids:

```python
iqae_epsilons = [0.05, 0.03, 0.02, 0.01, 0.007, 0.005]
iqae_alpha = 0.01
mlqae_eval_qubits = [2, 3, 4, 5, 6]
fae_maxiters = [3, 4, 5, 6, 7]
fae_delta = 0.05
```

Add loop blocks using `QuantumCalculator(..., extraction_mode="qae_iqae"|"qae_mlqae"|"qae_fae")` and append:

```python
points["IQAE"].append((oracle_calls, _mean_relative_error(estimate, exact)))
points["ML-QAE"].append((oracle_calls, _mean_relative_error(estimate, exact)))
points["FAE"].append((oracle_calls, _mean_relative_error(estimate, exact)))
```

Keep existing `for epsilon in epsilons` + `_min_calls_reaching_epsilon` + fallback pipeline unchanged, but iterate methods as:

```python
for method in ["Classical MC", "IQAE", "ML-QAE", "FAE"]:
```

**Step 4: Run targeted tests**

Run:

`uv run pytest tests/double_quant/application/test_risk_artifacts.py::test_generate_artifacts_equal_error_mentions_four_methods -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/risk/generate_artifacts.py tests/double_quant/application/test_risk_artifacts.py
git commit -m "feat(risk): add iqae and fae to equal-error benchmark"
```

### Task 3: Update equal-error plot to render 4 methods

**Files:**
- Modify: `scripts/risk/plot_from_artifacts.py`
- Test: `tests/double_quant/application/test_risk_artifacts.py`

**Step 1: Write the failing test**

Add a source-level plotting contract test.

```python
def test_plot_from_artifacts_equal_error_has_four_method_labels():
    content = Path("scripts/risk/plot_from_artifacts.py").read_text()
    assert '"Classical MC"' in content
    assert '"IQAE"' in content
    assert '"ML-QAE"' in content
    assert '"FAE"' in content
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/double_quant/application/test_risk_artifacts.py::test_plot_from_artifacts_equal_error_has_four_method_labels -v`
Expected: FAIL because current plot iterates only `Classical MC` and `ML-QAE`.

**Step 3: Write minimal implementation**

In `_plot_equal_error`, update palette and method iteration:

```python
palette = {
    "Classical MC": "#377eb8",
    "IQAE": "#4daf4a",
    "ML-QAE": "#ff7f00",
    "FAE": "#984ea3",
}

for method in ["Classical MC", "IQAE", "ML-QAE", "FAE"]:
    ...
```

Keep output filename unchanged (`equal_error_oracle_calls_fixed_grid_fallback.png`).

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/double_quant/application/test_risk_artifacts.py::test_plot_from_artifacts_equal_error_has_four_method_labels -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/risk/plot_from_artifacts.py tests/double_quant/application/test_risk_artifacts.py
git commit -m "feat(risk): plot four-method equal-error oracle curves"
```

### Task 4: Regenerate artifacts and validate outputs end-to-end

**Files:**
- Regenerate: `docs/assets/risk/data/*.csv`, `docs/assets/risk/*.png`, `docs/assets/risk/data/manifest.json`
- Verify references: `experiments/risk/artifacts.py`

**Step 1: Run artifact generation with overwrite**

Run: `uv run scripts/risk/generate_artifacts.py --force`
Expected:
- `quantum_comparison_n{3,4,5,6}.csv` rewritten with updated IQAE settings
- `equal_error_oracle_calls_summary.csv` rewritten with 4 methods
- `manifest.json` rewritten

**Step 2: Validate summary CSV contains 4 methods**

Run:

`uv run python -c "import pandas as pd; df=pd.read_csv('docs/assets/risk/data/equal_error_oracle_calls_summary.csv'); print(sorted(df['method'].unique()))"`

Expected output includes: `['Classical MC', 'FAE', 'IQAE', 'ML-QAE']`.

**Step 3: Render updated figures**

Run: `uv run scripts/risk/plot_from_artifacts.py`
Expected: `docs/assets/risk/equal_error_oracle_calls_fixed_grid_fallback.png` refreshed with 4 legend entries.

**Step 4: Run focused regression tests**

Run: `uv run pytest tests/double_quant/application/test_risk_artifacts.py -v`
Expected: PASS.

**Step 5: Final commit**

```bash
git add docs/assets/risk/data docs/assets/risk scripts/risk tests/double_quant/application/test_risk_artifacts.py
git commit -m "feat(risk): compare oracle calls at equal error across four methods"
```

### Task 5: Documentation sync for reproducibility metadata

**Files:**
- Modify: `scripts/risk/generate_artifacts.py`
- Optional verify: `docs/application/risk-experiment-workflow.md`

**Step 1: Write failing test for manifest parameter block visibility**

```python
def test_generate_artifacts_manifest_includes_equal_error_config_keys():
    content = Path("scripts/risk/generate_artifacts.py").read_text()
    assert '"equal_error"' in content
    assert '"epsilons"' in content
```

**Step 2: Run test to confirm behavior**

Run: `uv run pytest tests/double_quant/application/test_risk_artifacts.py::test_generate_artifacts_manifest_includes_equal_error_config_keys -v`
Expected: PASS if already present; if failing, proceed with Step 3.

**Step 3: Ensure manifest captures new method grids**

In the manifest `params["equal_error"]`, include new grids (`iqae_epsilons`, `iqae_alpha`, `mlqae_eval_qubits`, `fae_maxiters`, `fae_delta`) so runs are reproducible.

**Step 4: Re-run test and quick smoke generation**

Run:

- `uv run pytest tests/double_quant/application/test_risk_artifacts.py::test_generate_artifacts_manifest_includes_equal_error_config_keys -v`
- `uv run scripts/risk/generate_artifacts.py --force`

Expected: PASS + manifest contains new keys.

**Step 5: Commit**

```bash
git add scripts/risk/generate_artifacts.py docs/assets/risk/data/manifest.json tests/double_quant/application/test_risk_artifacts.py
git commit -m "chore(risk): record equal-error method grids in manifest"
```

## Verification Checklist

- `equal_error_oracle_calls_summary.csv` has all four methods.
- Equal-error figure contains four visible curves.
- IQAE setting in quantum comparison is `epsilon=0.01, alpha=0.01`.
- Existing required snapshot file list remains valid (`experiments/risk/artifacts.py`).
- Targeted risk artifact tests pass.

## Notes

- Follow `@superpowers:test-driven-development` for each code-change task.
- Before declaring completion, run `@superpowers:verification-before-completion`.
