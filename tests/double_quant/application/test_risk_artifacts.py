import json
from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.risk.artifacts import (
    REQUIRED_SNAPSHOT_FILES,
    get_artifact_paths,
    require_snapshot_files,
    write_manifest,
)


def test_default_paths_match_design():
    paths = get_artifact_paths()
    assert str(paths.cache_dir) == "tests/double_quant/application/cache"
    assert str(paths.snapshot_dir) == "docs/assets/risk/data"
    assert str(paths.figure_dir) == "docs/assets/risk"


def test_require_snapshot_files_raises_when_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        require_snapshot_files(tmp_path, REQUIRED_SNAPSHOT_FILES)


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


def test_test_risk_contains_no_savefig_calls():
    content = Path("tests/double_quant/application/test_risk.py").read_text()
    assert "savefig(" not in content


def test_application_gitignore_ignores_cache():
    content = Path("tests/double_quant/application/.gitignore").read_text()
    assert "cache" in {line.strip() for line in content.splitlines()}


def test_workflow_doc_contains_uv_run_commands():
    path = Path("docs/application/risk-experiment-workflow.md")
    assert path.exists()
    content = path.read_text()
    assert "uv run scripts/risk/generate_artifacts.py" in content
    assert "uv run scripts/risk/plot_from_artifacts.py" in content


def test_experiment_artifact_module_is_outside_src():
    assert Path("experiments/risk/artifacts.py").exists()
    assert not Path("src/double_quant/application/risk_artifacts.py").exists()


def test_artifact_module_has_no_legacy_cache_fallback():
    content = Path("experiments/risk/artifacts.py").read_text()
    assert (
        "tests/double_quant/application/data/experiment_data_clean.csv" not in content
    )


def test_quantum_comparison_method_labels_use_display_names():
    content = Path("scripts/risk/generate_artifacts.py").read_text()
    assert '"shots=1024"' in content
    assert '"shots=4096"' in content
    assert '"I-QAE"' in content
    assert '"F-QAE"' in content
    assert '"ML-QAE"' in content
    assert '"Statevector"' in content
