from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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


def test_quantum_comparison_uses_local_panel_y_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot_dir = tmp_path / "snapshot"
    figure_dir = tmp_path / "figure"
    snapshot_dir.mkdir()
    figure_dir.mkdir()

    panel_rel_errors = {
        3: [0.020, 0.025, 0.030],
        4: [0.040, 0.045, 0.050],
        5: [0.060, 0.065, 0.070],
        6: [1.0, 2.5, 4.0],
    }

    for n, rel_errors in panel_rel_errors.items():
        df = pd.DataFrame(
            {
                "n_l": [2, 3, 4],
                "method": ["qae_fae", "qae_fae", "qae_fae"],
                "rel_error": rel_errors,
            }
        )
        df.to_csv(snapshot_dir / f"quantum_comparison_n{n}.csv", index=False)

    captured: dict[str, Any] = {}
    original_subplots = plt.subplots
    original_close = plt.close

    def _capture_subplots(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        fig, axes = original_subplots(*args, **kwargs)
        captured["fig"] = fig
        captured["axes"] = axes
        return fig, axes

    monkeypatch.setattr(plt, "subplots", _capture_subplots)
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)

    _plot_quantum_comparison(str(snapshot_dir), str(figure_dir))

    axes = captured["axes"].flatten()
    y_limits_by_n: dict[int, tuple[float, float]] = {}
    for ax in axes:
        title = ax.get_title()
        assert "local y-scale" in title
        n_value = int(title.split("=")[1].split()[0])
        y_limits_by_n[n_value] = ax.get_ylim()

    assert set(y_limits_by_n) == {3, 4, 5, 6}
    assert len(set(y_limits_by_n.values())) > 1
    n3_ylim = y_limits_by_n[3]
    n6_ylim = y_limits_by_n[6]
    assert n6_ylim[1] - n6_ylim[0] > 10 * (n3_ylim[1] - n3_ylim[0])

    original_close(captured["fig"])
