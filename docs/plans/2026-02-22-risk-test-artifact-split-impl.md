# Risk Test/Plot Split Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Decouple heavy artifact generation and plotting from `pytest` so style changes only require re-plotting, while keeping final figures versioned.

**Architecture:** Keep `tests/double_quant/application/test_risk.py` focused on correctness assertions only. Move heavy experiment execution into `scripts/risk/generate_artifacts.py` and move rendering into `scripts/risk/plot_from_artifacts.py`. Store non-versioned cache under `tests/double_quant/application/cache/`, versioned snapshot data under `docs/assets/risk/data/`, and versioned final figures under `docs/assets/risk/`.

**Tech Stack:** Python 3.11, pytest, pandas, numpy, matplotlib, seaborn, uv

---

### Task 1: Add artifact path and validation helpers

**Files:**
- Create: `src/double_quant/application/risk_artifacts.py`
- Create: `tests/double_quant/application/test_risk_artifacts.py`

**Step 1: Write the failing test**

```python
from pathlib import Path

import pytest

from double_quant.application.risk_artifacts import (
    REQUIRED_SNAPSHOT_FILES,
    get_artifact_paths,
    require_snapshot_files,
)


def test_default_paths_match_design():
    paths = get_artifact_paths()
    assert str(paths.cache_dir) == "tests/double_quant/application/cache"
    assert str(paths.snapshot_dir) == "docs/assets/risk/data"
    assert str(paths.figure_dir) == "docs/assets/risk"


def test_require_snapshot_files_raises_when_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        require_snapshot_files(tmp_path, REQUIRED_SNAPSHOT_FILES)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/double_quant/application/test_risk_artifacts.py -v`
Expected: FAIL with `ModuleNotFoundError` for `double_quant.application.risk_artifacts`

**Step 3: Write minimal implementation**

```python
from dataclasses import dataclass
from pathlib import Path


REQUIRED_SNAPSHOT_FILES = [
    "vol_buckets_metrics.csv",
    "vol_buckets_series.csv",
    "restoration_accuracy.csv",
    "equal_error_oracle_calls_summary.csv",
]


@dataclass(frozen=True)
class ArtifactPaths:
    cache_dir: Path
    snapshot_dir: Path
    figure_dir: Path


def get_artifact_paths() -> ArtifactPaths:
    return ArtifactPaths(
        cache_dir=Path("tests/double_quant/application/cache"),
        snapshot_dir=Path("docs/assets/risk/data"),
        figure_dir=Path("docs/assets/risk"),
    )


def require_snapshot_files(snapshot_dir: Path, file_names: list[str]) -> None:
    missing = [name for name in file_names if not (snapshot_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing snapshot files in {snapshot_dir}: {', '.join(sorted(missing))}"
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/double_quant/application/test_risk_artifacts.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/double_quant/application/risk_artifacts.py tests/double_quant/application/test_risk_artifacts.py
git commit -m "feat(application): add risk artifact path and snapshot validators"
```

---

### Task 2: Refactor `test_risk.py` to correctness-only tests

**Files:**
- Modify: `tests/double_quant/application/test_risk.py`

**Step 1: Write the failing test (guard against plot side effects)**

Add to `tests/double_quant/application/test_risk_artifacts.py`:

```python
from pathlib import Path


def test_test_risk_contains_no_savefig_calls():
    content = Path("tests/double_quant/application/test_risk.py").read_text()
    assert "savefig(" not in content
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/double_quant/application/test_risk_artifacts.py::test_test_risk_contains_no_savefig_calls -v`
Expected: FAIL because `savefig(` currently exists in `test_risk.py`

**Step 3: Write minimal implementation**

- Remove plotting code from:
  - `test_volatility_bucketing`
  - `TestRiskSaving.test_restoration_accuracy`
- Remove heavy benchmark test methods from `TestQuantumPerformance`.
- Delete `test_quantum_vs_classical_mc` completely.
- Keep/adjust assertions so correctness tests still pass.

**Step 4: Run target tests**

Run: `uv run pytest tests/double_quant/application/test_risk.py::test_volatility_bucketing -v`
Expected: PASS (assertion-only)

Run: `uv run pytest tests/double_quant/application/test_risk.py::TestRiskSaving::test_restoration_accuracy -v`
Expected: PASS (numerical check only)

Run: `uv run pytest tests/double_quant/application/test_risk_artifacts.py::test_test_risk_contains_no_savefig_calls -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/double_quant/application/test_risk.py tests/double_quant/application/test_risk_artifacts.py
git commit -m "refactor(test): keep risk tests assertion-only and drop plot benchmarks"
```

---

### Task 3: Add generation script for versioned snapshot data

**Files:**
- Create: `scripts/risk/generate_artifacts.py`
- Modify: `tests/double_quant/application/test_risk.py` (reuse helper code if needed)

**Step 1: Write the failing test (manifest contract)**

Add to `tests/double_quant/application/test_risk_artifacts.py`:

```python
import json
from pathlib import Path

from double_quant.application.risk_artifacts import write_manifest


def test_manifest_contains_required_keys(tmp_path: Path):
    path = write_manifest(
        output_dir=tmp_path,
        params={"n_rounds": 8},
        source_data="tests/double_quant/application/cache/experiment_data_clean.csv",
    )
    payload = json.loads(path.read_text())
    assert "generated_at" in payload
    assert "params" in payload
    assert "source_data" in payload
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/double_quant/application/test_risk_artifacts.py::test_manifest_contains_required_keys -v`
Expected: FAIL with missing `write_manifest`

**Step 3: Write minimal implementation**

- Add `write_manifest(...)` to `src/double_quant/application/risk_artifacts.py`.
- Implement `scripts/risk/generate_artifacts.py` to:
  - load/download base prices via cache path `tests/double_quant/application/cache/`
  - compute and write snapshot CSV files into `docs/assets/risk/data/`
  - write/update `manifest.json`
  - support `--force` to overwrite existing snapshot files

**Step 4: Run tests and dry-run command**

Run: `uv run pytest tests/double_quant/application/test_risk_artifacts.py::test_manifest_contains_required_keys -v`
Expected: PASS

Run: `uv run scripts/risk/generate_artifacts.py --help`
Expected: prints CLI usage with `--force`

**Step 5: Commit**

```bash
git add src/double_quant/application/risk_artifacts.py scripts/risk/generate_artifacts.py tests/double_quant/application/test_risk_artifacts.py
git commit -m "feat(risk): add artifact generation script and manifest writer"
```

---

### Task 4: Add plotting script that reads snapshots only

**Files:**
- Create: `scripts/risk/plot_from_artifacts.py`
- Modify: `src/double_quant/application/risk_artifacts.py`

**Step 1: Write the failing test**

Add to `tests/double_quant/application/test_risk_artifacts.py`:

```python
from pathlib import Path

import pytest

from double_quant.application.risk_artifacts import (
    REQUIRED_SNAPSHOT_FILES,
    require_snapshot_files,
)


def test_plot_requires_snapshot_files(tmp_path: Path):
    (tmp_path / REQUIRED_SNAPSHOT_FILES[0]).write_text("dummy")
    with pytest.raises(FileNotFoundError):
        require_snapshot_files(tmp_path, REQUIRED_SNAPSHOT_FILES)
```

**Step 2: Run test to verify it fails (if helper still permissive)**

Run: `uv run pytest tests/double_quant/application/test_risk_artifacts.py::test_plot_requires_snapshot_files -v`
Expected: FAIL before strict missing-file enforcement is added

**Step 3: Write minimal implementation**

- Ensure `require_snapshot_files(...)` checks all required files.
- Implement `scripts/risk/plot_from_artifacts.py` to:
  - verify required snapshot files exist
  - load snapshot CSV files from `docs/assets/risk/data/`
  - generate PNG files into `docs/assets/risk/`
  - never call heavy compute/download logic

**Step 4: Run tests and dry-run command**

Run: `uv run pytest tests/double_quant/application/test_risk_artifacts.py::test_plot_requires_snapshot_files -v`
Expected: PASS

Run: `uv run scripts/risk/plot_from_artifacts.py --help`
Expected: prints CLI usage and snapshot path info

**Step 5: Commit**

```bash
git add scripts/risk/plot_from_artifacts.py src/double_quant/application/risk_artifacts.py tests/double_quant/application/test_risk_artifacts.py
git commit -m "feat(risk): add snapshot-driven plotting script"
```

---

### Task 5: Update cache ignore rules and data location defaults

**Files:**
- Modify: `tests/double_quant/application/.gitignore`
- Modify: `tests/double_quant/application/test_risk.py`

**Step 1: Write the failing test**

Add to `tests/double_quant/application/test_risk_artifacts.py`:

```python
from pathlib import Path


def test_application_gitignore_ignores_cache():
    content = Path("tests/double_quant/application/.gitignore").read_text()
    assert "cache" in {line.strip() for line in content.splitlines()}
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/double_quant/application/test_risk_artifacts.py::test_application_gitignore_ignores_cache -v`
Expected: FAIL when `cache` entry is absent

**Step 3: Write minimal implementation**

- Update `tests/double_quant/application/.gitignore` to include `cache`.
- Update `DataPreparation` default cache path in `test_risk.py` to `tests/double_quant/application/cache/experiment_data_clean.csv`.

**Step 4: Run tests**

Run: `uv run pytest tests/double_quant/application/test_risk_artifacts.py::test_application_gitignore_ignores_cache -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/double_quant/application/.gitignore tests/double_quant/application/test_risk.py tests/double_quant/application/test_risk_artifacts.py
git commit -m "chore(test): switch risk cache location to application/cache"
```

---

### Task 6: Document the workflow outside README

**Files:**
- Create: `docs/application/risk-experiment-workflow.md`

**Step 1: Write the failing doc check test**

Add to `tests/double_quant/application/test_risk_artifacts.py`:

```python
from pathlib import Path


def test_workflow_doc_contains_uv_run_commands():
    path = Path("docs/application/risk-experiment-workflow.md")
    assert path.exists()
    content = path.read_text()
    assert "uv run scripts/risk/generate_artifacts.py" in content
    assert "uv run scripts/risk/plot_from_artifacts.py" in content
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/double_quant/application/test_risk_artifacts.py::test_workflow_doc_contains_uv_run_commands -v`
Expected: FAIL with missing file

**Step 3: Write minimal implementation**

Create `docs/application/risk-experiment-workflow.md` with:

- three-layer artifact layout (`cache`, `docs/assets/risk/data`, `docs/assets/risk`)
- two command entries (`uv run scripts/risk/generate_artifacts.py`, `uv run scripts/risk/plot_from_artifacts.py`)
- error handling notes (`missing snapshots`, `--force`)

**Step 4: Run tests**

Run: `uv run pytest tests/double_quant/application/test_risk_artifacts.py::test_workflow_doc_contains_uv_run_commands -v`
Expected: PASS

**Step 5: Commit**

```bash
git add docs/application/risk-experiment-workflow.md tests/double_quant/application/test_risk_artifacts.py
git commit -m "docs(risk): add experiment workflow for data generation and plotting"
```

---

## Final Verification

Run:

```bash
uv run pytest tests/double_quant/application/test_risk_artifacts.py -v
uv run pytest tests/double_quant/application/test_risk.py::test_volatility_bucketing -v
uv run pytest tests/double_quant/application/test_risk.py::TestRiskSaving::test_restoration_accuracy -v
uv run scripts/risk/generate_artifacts.py --help
uv run scripts/risk/plot_from_artifacts.py --help
```

Expected:

1. Artifact utility tests PASS
2. Kept correctness tests PASS
3. Script CLIs are discoverable and do not trigger unintended work on `--help`
