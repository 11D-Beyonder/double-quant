from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace

from double_quant.algorithm.circuit.repair import repair_quantum_circuit

BlochVector = tuple[float, float, float]

_DISPLAY_OPERATION_NAMES = {
    "State Preparation": "状态制备",
    "state_preparation": "状态制备",
    "QPE": "相位估计",
    "QPE_dg": "逆相位估计",
    "ucry": "风险旋转",
    "cry": "受控旋转",
    "ry": "旋转门",
    "h": "H门",
    "cx": "受控非门",
}


@dataclass(frozen=True, slots=True)
class StateEvolutionStep:
    index: int
    label: str
    statevector: np.ndarray
    probabilities: Mapping[str, float]
    bloch_vectors: Mapping[int, BlochVector]


@dataclass(frozen=True, slots=True)
class StateEvolutionVisualization:
    circuit: QuantumCircuit
    steps: tuple[StateEvolutionStep, ...]
    figure: Figure
    image_path: Path | None
    animation_path: Path | None


@dataclass(frozen=True, slots=True)
class CircuitVisualization:
    circuit: QuantumCircuit
    figure: Figure
    text_diagram: str
    image_path: Path | None


@dataclass(frozen=True, slots=True)
class ComputationProcessVisualization:
    circuit: QuantumCircuit
    steps: tuple[StateEvolutionStep, ...]
    operation_labels: tuple[str, ...]
    final_probabilities: Mapping[str, float]
    figure: Figure
    image_path: Path | None
    animation_path: Path | None


class CircuitVisualizationError(ValueError):
    """Raised when a circuit cannot be converted into a visualizable form."""


def visualize_state_evolution(
    circuit: QuantumCircuit,
    *,
    output_path: str | Path | None = None,
    animation_path: str | Path | None = None,
    title: str = "量子态演化",
    tracked_qubits: Sequence[int] | None = None,
    max_bloch_qubits: int = 3,
    max_basis_states: int = 16,
    fps: int = 1,
    dpi: int = 160,
) -> StateEvolutionVisualization:
    """Visualize state evolution with probability bars, Bloch spheres, and GIF."""

    prepared = _prepare_statevector_circuit(circuit)
    steps = _state_evolution_steps(prepared)
    tracked = _normalize_tracked_qubits(
        tracked_qubits,
        num_qubits=prepared.num_qubits,
        max_bloch_qubits=max_bloch_qubits,
    )

    _configure_chinese_font()
    figure = plt.figure(figsize=(12, 6), dpi=dpi)
    _plot_state_snapshot(
        figure,
        steps[-1],
        num_qubits=prepared.num_qubits,
        tracked_qubits=tracked,
        title=title,
        max_basis_states=max_basis_states,
    )
    image_path = _save_figure(figure, output_path, dpi)
    gif_path = _save_state_animation(
        steps,
        num_qubits=prepared.num_qubits,
        tracked_qubits=tracked,
        animation_path=animation_path,
        title=title,
        max_basis_states=max_basis_states,
        fps=fps,
        dpi=dpi,
    )
    return StateEvolutionVisualization(
        circuit=prepared,
        steps=steps,
        figure=figure,
        image_path=image_path,
        animation_path=gif_path,
    )


def visualize_quantum_circuit(
    circuit: QuantumCircuit,
    *,
    output_path: str | Path | None = None,
    title: str = "量子电路",
    fold: int = -1,
    scale: float = 0.8,
    style: dict[str, Any] | None = None,
    decompose_reps: int = 0,
    dpi: int = 160,
) -> CircuitVisualization:
    """Render a Qiskit circuit as both a Matplotlib figure and text diagram."""

    _configure_chinese_font()
    visual_circuit = _decompose_circuit(circuit, decompose_reps)
    _apply_display_operation_labels(visual_circuit)
    drawing = visual_circuit.draw(output="mpl", fold=fold, scale=scale, style=style)
    if not isinstance(drawing, Figure):
        raise CircuitVisualizationError("Qiskit did not return a Matplotlib figure.")
    drawing.suptitle(title)
    text_diagram = str(visual_circuit.draw(output="text", fold=fold))
    image_path = _save_figure(drawing, output_path, dpi)
    return CircuitVisualization(
        circuit=visual_circuit,
        figure=drawing,
        text_diagram=text_diagram,
        image_path=image_path,
    )


def visualize_quantum_computation_process(
    circuit: QuantumCircuit,
    *,
    output_path: str | Path | None = None,
    animation_path: str | Path | None = None,
    title: str = "量子计算过程",
    tracked_qubits: Sequence[int] | None = None,
    max_bloch_qubits: int = 3,
    max_basis_states: int = 16,
    fps: int = 1,
    dpi: int = 160,
) -> ComputationProcessVisualization:
    """Visualize gate order, state evolution, final probabilities, and GIF."""

    prepared = _prepare_statevector_circuit(circuit)
    steps = _state_evolution_steps(prepared)
    operation_labels = _operation_labels(prepared)
    tracked = _normalize_tracked_qubits(
        tracked_qubits,
        num_qubits=prepared.num_qubits,
        max_bloch_qubits=max_bloch_qubits,
    )

    _configure_chinese_font()
    figure = plt.figure(figsize=(12, 8), dpi=dpi)
    _plot_process_snapshot(
        figure,
        prepared,
        steps[-1],
        tracked_qubits=tracked,
        title=title,
        max_basis_states=max_basis_states,
    )
    image_path = _save_figure(figure, output_path, dpi)
    gif_path = _save_process_animation(
        prepared,
        steps,
        tracked_qubits=tracked,
        animation_path=animation_path,
        title=title,
        max_basis_states=max_basis_states,
        fps=fps,
        dpi=dpi,
    )
    return ComputationProcessVisualization(
        circuit=prepared,
        steps=steps,
        operation_labels=operation_labels,
        final_probabilities=steps[-1].probabilities,
        figure=figure,
        image_path=image_path,
        animation_path=gif_path,
    )


def _prepare_statevector_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    if circuit.num_qubits <= 0:
        raise CircuitVisualizationError("Circuit must contain at least one qubit.")
    if "measure" not in circuit.count_ops():
        return circuit.copy()
    try:
        return repair_quantum_circuit(circuit, mode="statevector").circuit
    except ValueError as exc:
        raise CircuitVisualizationError(str(exc)) from exc


def _configure_chinese_font() -> None:
    preferred = [
        "Microsoft YaHei",
        "PingFang SC",
        "Hiragino Sans GB",
        "Hiragino Sans",
        "Heiti SC",
        "STHeiti",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def _decompose_circuit(circuit: QuantumCircuit, decompose_reps: int) -> QuantumCircuit:
    if decompose_reps < 0:
        raise CircuitVisualizationError("decompose_reps must be non-negative.")
    visual_circuit = circuit.copy()
    for _ in range(decompose_reps):
        visual_circuit = visual_circuit.decompose()
    return visual_circuit


def _apply_display_operation_labels(circuit: QuantumCircuit) -> None:
    for instruction in circuit.data:
        operation = instruction.operation
        display_name = _display_instruction_name(operation.name)
        if display_name != operation.name:
            operation.label = display_name


def _state_evolution_steps(
    circuit: QuantumCircuit,
) -> tuple[StateEvolutionStep, ...]:
    prefix = QuantumCircuit(circuit.num_qubits)
    state = Statevector.from_int(0, 2**circuit.num_qubits)
    steps = [
        _state_step(
            index=0,
            label="initial |0...0>",
            state=state,
            num_qubits=circuit.num_qubits,
        )
    ]

    for index, instruction in enumerate(circuit.data, start=1):
        operation = instruction.operation
        name = operation.name
        if instruction.clbits:
            raise CircuitVisualizationError(
                f"Instruction '{name}' uses classical bits and cannot be visualized "
                "with statevector evolution."
            )

        if name in {"barrier", "delay"}:
            label = f"{index}: {name}"
        elif name in {"measure", "reset"}:
            raise CircuitVisualizationError(
                f"Instruction '{name}' is not supported for statevector visualization."
            )
        else:
            qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
            prefix.append(operation.copy(), qubits)
            label = _instruction_label(index, name, qubits)

        state = Statevector.from_instruction(prefix)
        steps.append(
            _state_step(
                index=index,
                label=label,
                state=state,
                num_qubits=circuit.num_qubits,
            )
        )

    return tuple(steps)


def _state_step(
    *,
    index: int,
    label: str,
    state: Statevector,
    num_qubits: int,
) -> StateEvolutionStep:
    probabilities = _probability_distribution(state, num_qubits)
    return StateEvolutionStep(
        index=index,
        label=label,
        statevector=np.asarray(state.data, dtype=complex).copy(),
        probabilities=probabilities,
        bloch_vectors=_bloch_vectors(state, num_qubits),
    )


def _probability_distribution(
    state: Statevector,
    num_qubits: int,
) -> dict[str, float]:
    labels = _basis_labels(num_qubits)
    probabilities = np.asarray(state.probabilities(), dtype=float)
    return {
        label: float(probability)
        for label, probability in zip(labels, probabilities, strict=True)
        if probability > 1e-12
    }


def _bloch_vectors(state: Statevector, num_qubits: int) -> dict[int, BlochVector]:
    density = DensityMatrix(state)
    vectors: dict[int, BlochVector] = {}
    for qubit in range(num_qubits):
        traced_out = [index for index in range(num_qubits) if index != qubit]
        reduced = partial_trace(density, traced_out) if traced_out else density
        rho = np.asarray(reduced.data, dtype=complex)
        vectors[qubit] = (
            float(2 * np.real(rho[0, 1])),
            float(-2 * np.imag(rho[0, 1])),
            float(np.real(rho[0, 0] - rho[1, 1])),
        )
    return vectors


def _plot_state_snapshot(
    figure: Figure,
    step: StateEvolutionStep,
    *,
    num_qubits: int,
    tracked_qubits: tuple[int, ...],
    title: str,
    max_basis_states: int,
) -> None:
    figure.clear()
    columns = max(1, len(tracked_qubits))
    grid = figure.add_gridspec(2, columns, height_ratios=[1.15, 1.0])

    probability_axis = figure.add_subplot(grid[0, :])
    _plot_probability_distribution(
        probability_axis,
        step,
        num_qubits=num_qubits,
        max_basis_states=max_basis_states,
        title=f"{_display_step_label(step.label)} - 基态概率",
    )

    for index, qubit in enumerate(tracked_qubits):
        bloch_axis = figure.add_subplot(grid[1, index], projection="3d")
        _plot_bloch_sphere(
            bloch_axis,
            step.bloch_vectors[qubit],
            title=f"q[{qubit}] 布洛赫向量",
        )

    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.94))


def _plot_process_snapshot(
    figure: Figure,
    circuit: QuantumCircuit,
    step: StateEvolutionStep,
    *,
    tracked_qubits: tuple[int, ...],
    title: str,
    max_basis_states: int,
) -> None:
    figure.clear()
    columns = max(1, len(tracked_qubits))
    grid = figure.add_gridspec(3, columns, height_ratios=[0.9, 1.0, 1.0])

    timeline_axis = figure.add_subplot(grid[0, :])
    _plot_gate_timeline(timeline_axis, circuit, active_step=step.index)

    probability_axis = figure.add_subplot(grid[1, :])
    _plot_probability_distribution(
        probability_axis,
        step,
        num_qubits=circuit.num_qubits,
        max_basis_states=max_basis_states,
        title=f"{_display_step_label(step.label)} - 基态概率",
    )

    for index, qubit in enumerate(tracked_qubits):
        bloch_axis = figure.add_subplot(grid[2, index], projection="3d")
        _plot_bloch_sphere(
            bloch_axis,
            step.bloch_vectors[qubit],
            title=f"q[{qubit}] 布洛赫向量",
        )

    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.95))


def _plot_probability_distribution(
    axis: Axes,
    step: StateEvolutionStep,
    *,
    num_qubits: int,
    max_basis_states: int,
    title: str,
) -> None:
    labels = _basis_labels(num_qubits)
    probabilities = _probability_array(step, num_qubits)
    visible_columns = _visible_basis_columns(
        probabilities.reshape(1, -1),
        max_basis_states,
    )
    visible_labels = [labels[index] for index in visible_columns]
    visible_probabilities = probabilities[visible_columns]

    axis.bar(visible_labels, visible_probabilities, color="#2f5597")
    axis.set_title(title)
    axis.set_xlabel("基态")
    axis.set_ylabel("概率")
    axis.set_ylim(0, max(1.0, float(visible_probabilities.max()) * 1.15))
    axis.tick_params(axis="x", rotation=45)
    axis.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)


def _plot_bloch_sphere(axis: Axes, vector: BlochVector, *, title: str) -> None:
    axis3d = cast(Any, axis)
    u = np.linspace(0, np.pi, 28)
    v = np.linspace(0, 2 * np.pi, 56)
    x = np.outer(np.sin(u), np.cos(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.cos(u), np.ones_like(v))

    axis3d.plot_surface(x, y, z, color="#d9e2f3", alpha=0.16, linewidth=0)
    axis3d.plot_wireframe(x, y, z, color="#8faadc", alpha=0.22, linewidth=0.45)
    axis3d.quiver(0, 0, 0, 1, 0, 0, color="#c00000", linewidth=1.0, alpha=0.45)
    axis3d.quiver(0, 0, 0, 0, 1, 0, color="#70ad47", linewidth=1.0, alpha=0.45)
    axis3d.quiver(0, 0, 0, 0, 0, 1, color="#4472c4", linewidth=1.0, alpha=0.45)
    axis3d.quiver(
        0,
        0,
        0,
        vector[0],
        vector[1],
        vector[2],
        color="#7030a0",
        linewidth=2.5,
        arrow_length_ratio=0.16,
    )
    axis3d.scatter([vector[0]], [vector[1]], [vector[2]], color="#7030a0", s=32)

    axis.set_title(title)
    axis.set_xlim(-1.05, 1.05)
    axis.set_ylim(-1.05, 1.05)
    axis3d.set_zlim(-1.05, 1.05)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis3d.set_zlabel("Z")
    axis3d.set_box_aspect((1, 1, 1))


def _plot_gate_timeline(
    axis: Axes,
    circuit: QuantumCircuit,
    *,
    active_step: int | None = None,
) -> None:
    operations = list(circuit.data)
    axis.set_title("算法门时间线")
    axis.set_xlabel("操作序号")
    axis.set_ylabel("量子比特")
    axis.set_yticks(
        range(circuit.num_qubits),
        [f"q[{index}]" for index in range(circuit.num_qubits)],
    )
    axis.set_ylim(circuit.num_qubits - 0.5, -0.5)

    if not operations:
        axis.text(0.5, 0.5, "空电路", ha="center", va="center")
        axis.set_xticks([])
        return

    axis.set_xlim(0.5, len(operations) + 0.5)
    axis.set_xticks(range(1, len(operations) + 1))
    for index, instruction in enumerate(operations, start=1):
        qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        if not qubits:
            continue
        color = "#c00000" if active_step == index else "#2f5597"
        axis.vlines(index, min(qubits), max(qubits), color="#44546a", linewidth=1.2)
        axis.scatter(
            [index] * len(qubits),
            qubits,
            marker="s",
            s=170,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        label_y = min(max(qubits) + 0.28, circuit.num_qubits - 0.05)
        axis.text(
            index,
            label_y,
            _display_instruction_name(instruction.operation.name),
            ha="center",
            va="bottom",
            rotation=55,
            fontsize=8,
        )
    axis.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.5)


def _display_instruction_name(name: str) -> str:
    return _DISPLAY_OPERATION_NAMES.get(name, name)


def _display_step_label(label: str) -> str:
    if label == "initial |0...0>":
        return "初始态 |0...0>"
    prefix, separator, rest = label.partition(": ")
    if not separator:
        return _display_instruction_name(label)
    name, bracket, suffix = rest.partition("[")
    display_name = _display_instruction_name(name)
    if bracket:
        return f"{prefix}: {display_name}[{suffix}"
    return f"{prefix}: {display_name}"


def _save_state_animation(
    steps: tuple[StateEvolutionStep, ...],
    *,
    num_qubits: int,
    tracked_qubits: tuple[int, ...],
    animation_path: str | Path | None,
    title: str,
    max_basis_states: int,
    fps: int,
    dpi: int,
) -> Path | None:
    if animation_path is None:
        return None
    path = Path(animation_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(12, 6), dpi=dpi)

    def update(frame_index: int) -> None:
        _plot_state_snapshot(
            figure,
            steps[frame_index],
            num_qubits=num_qubits,
            tracked_qubits=tracked_qubits,
            title=title,
            max_basis_states=max_basis_states,
        )

    animation = FuncAnimation(figure, update, frames=len(steps), repeat=True)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(figure)
    return path


def _save_process_animation(
    circuit: QuantumCircuit,
    steps: tuple[StateEvolutionStep, ...],
    *,
    tracked_qubits: tuple[int, ...],
    animation_path: str | Path | None,
    title: str,
    max_basis_states: int,
    fps: int,
    dpi: int,
) -> Path | None:
    if animation_path is None:
        return None
    path = Path(animation_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(12, 8), dpi=dpi)

    def update(frame_index: int) -> None:
        _plot_process_snapshot(
            figure,
            circuit,
            steps[frame_index],
            tracked_qubits=tracked_qubits,
            title=title,
            max_basis_states=max_basis_states,
        )

    animation = FuncAnimation(figure, update, frames=len(steps), repeat=True)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(figure)
    return path


def _probability_matrix(
    steps: tuple[StateEvolutionStep, ...],
    num_qubits: int,
) -> np.ndarray:
    return np.vstack([_probability_array(step, num_qubits) for step in steps])


def _probability_array(step: StateEvolutionStep, num_qubits: int) -> np.ndarray:
    labels = _basis_labels(num_qubits)
    return np.array([step.probabilities.get(label, 0.0) for label in labels])


def _visible_basis_columns(matrix: np.ndarray, max_basis_states: int) -> np.ndarray:
    if max_basis_states <= 0:
        raise CircuitVisualizationError("max_basis_states must be positive.")
    if matrix.shape[1] <= max_basis_states:
        return np.arange(matrix.shape[1])

    maxima = matrix.max(axis=0)
    ranked = np.argsort(maxima)[-max_basis_states:]
    return np.sort(ranked)


def _normalize_tracked_qubits(
    tracked_qubits: Sequence[int] | None,
    *,
    num_qubits: int,
    max_bloch_qubits: int,
) -> tuple[int, ...]:
    if max_bloch_qubits <= 0:
        raise CircuitVisualizationError("max_bloch_qubits must be positive.")
    if tracked_qubits is None:
        return tuple(range(min(num_qubits, max_bloch_qubits)))
    normalized = tuple(int(qubit) for qubit in tracked_qubits)
    if len(normalized) > max_bloch_qubits:
        raise CircuitVisualizationError(
            "tracked_qubits length cannot exceed max_bloch_qubits."
        )
    if len(set(normalized)) != len(normalized):
        raise CircuitVisualizationError("tracked_qubits cannot contain duplicates.")
    if any(qubit < 0 or qubit >= num_qubits for qubit in normalized):
        raise CircuitVisualizationError("tracked_qubits contains an invalid qubit.")
    return normalized


def _operation_labels(circuit: QuantumCircuit) -> tuple[str, ...]:
    labels: list[str] = []
    for index, instruction in enumerate(circuit.data, start=1):
        qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        labels.append(_instruction_label(index, instruction.operation.name, qubits))
    return tuple(labels)


def _instruction_label(index: int, name: str, qubits: list[int]) -> str:
    if not qubits:
        return f"{index}: {name}"
    qubit_text = ",".join(str(qubit) for qubit in qubits)
    return f"{index}: {name}[{qubit_text}]"


def _basis_labels(num_qubits: int) -> tuple[str, ...]:
    width = max(num_qubits, 1)
    return tuple(format(index, f"0{width}b") for index in range(2**num_qubits))


def _save_figure(
    figure: Figure,
    output_path: str | Path | None,
    dpi: int,
) -> Path | None:
    if output_path is None:
        return None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", dpi=dpi)
    return path
