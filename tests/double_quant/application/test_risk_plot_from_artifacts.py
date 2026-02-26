from pathlib import Path
import sys
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
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


def test_quantum_comparison_legend_order_and_colors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot_dir = tmp_path / "snapshot"
    figure_dir = tmp_path / "figure"
    snapshot_dir.mkdir()
    figure_dir.mkdir()

    ordered_methods = [
        "shots=1024",
        "shots=4096",
        "I-QAE",
        "F-QAE",
        "ML-QAE",
        "Statevector",
    ]
    for n in (3, 4, 5, 6):
        rows: list[dict[str, float | int | str]] = []
        for n_l in (2, 3, 4):
            for method_idx, method in enumerate(ordered_methods):
                rows.append(
                    {
                        "n_l": n_l,
                        "method": method,
                        "rel_error": 0.01 * (method_idx + 1) * (n_l + 1),
                    }
                )
        pd.DataFrame(rows).to_csv(
            snapshot_dir / f"quantum_comparison_n{n}.csv", index=False
        )

    captured: dict[str, Any] = {}
    original_legend = Figure.legend

    def _capture_legend(self: Figure, *args: Any, **kwargs: Any) -> Any:
        handles = kwargs.get("handles")
        if handles is None and args:
            handles = args[0]
        captured["labels"] = [handle.get_label() for handle in handles]
        captured["colors"] = {
            handle.get_label(): handle.get_color() for handle in handles
        }
        captured["ncol"] = kwargs.get("ncol")
        return original_legend(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "legend", _capture_legend)

    _plot_quantum_comparison(str(snapshot_dir), str(figure_dir))

    assert captured["labels"] == ordered_methods
    assert captured["ncol"] == 6
    assert mcolors.to_hex(captured["colors"]["I-QAE"]).lower() == "#f58518"
    assert mcolors.to_hex(captured["colors"]["Statevector"]).lower() == "#7e57c2"
    assert (
        mcolors.to_hex(captured["colors"]["I-QAE"]).lower()
        != mcolors.to_hex(captured["colors"]["Statevector"]).lower()
    )
