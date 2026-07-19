from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray

from double_quant.algorithm.rasengan import LinearConstraintBinaryProblem


@dataclass(frozen=True, slots=True)
class RasenganProblemInstance:
    problem: LinearConstraintBinaryProblem
    transition_basis: NDArray[np.int_]
    feasible_state: NDArray[np.int_]
    search_space_size: int
    feasible_states: tuple[tuple[int, ...], ...] | None = None


def antifraud_cycle_instance(
    *,
    cycle_count: int,
    penalty: float = 400.0,
) -> RasenganProblemInstance:
    """Build a flow-conserving suspicious-cycle selection instance."""
    if cycle_count < 2:
        raise ValueError("cycle_count must be at least 2")
    variable_count = 3 * cycle_count
    node_count = 1 + 2 * cycle_count
    incidence = np.zeros((node_count, variable_count), dtype=float)
    variable_names: list[str] = []
    for cycle in range(cycle_count):
        root = 0
        first = 1 + 2 * cycle
        second = first + 1
        edge_start = 3 * cycle
        edges = ((root, first), (first, second), (second, root))
        for local_index, (source, target) in enumerate(edges):
            column = edge_start + local_index
            incidence[source, column] += 1.0
            incidence[target, column] -= 1.0
        variable_names.extend(
            (
                f"cycle_{cycle}_root_to_a",
                f"cycle_{cycle}_a_to_b",
                f"cycle_{cycle}_b_to_root",
            )
        )
    cardinality = np.ones((1, variable_count), dtype=float)
    constraints = np.vstack([incidence, cardinality])
    rhs = np.zeros(constraints.shape[0], dtype=float)
    rhs[-1] = 3.0
    coefficients = np.asarray(
        [7.0 + 0.9 * cycle + 0.35 * local for cycle in range(cycle_count) for local in range(3)],
        dtype=float,
    )
    quadratic = _chain_quadratic(variable_count, -0.035)
    problem = LinearConstraintBinaryProblem(
        linear=coefficients,
        constraints=constraints,
        rhs=rhs,
        sense="max",
        quadratic=quadratic,
        penalty=penalty,
        variable_names=tuple(variable_names),
    )
    feasible_state = np.zeros(variable_count, dtype=int)
    feasible_state[:3] = 1
    transitions = []
    for cycle in range(1, cycle_count):
        move = np.zeros(variable_count, dtype=int)
        move[:3] = -1
        move[3 * cycle : 3 * cycle + 3] = 1
        transitions.append(move)
    return RasenganProblemInstance(
        problem=problem,
        transition_basis=np.asarray(transitions, dtype=int),
        feasible_state=feasible_state,
        search_space_size=cycle_count,
        feasible_states=tuple(tuple(state.tolist()) for state in _cycle_feasible_states(cycle_count)),
    )


def antifraud_cycle_choice_instance(
    *,
    cycle_lengths: tuple[int, ...],
    penalty: float = 400.0,
) -> RasenganProblemInstance:
    """Build mixed suspicious-cycle replacement blocks.

    Each block contains two alternative closed transaction cycles of the same
    length. Selecting either cycle satisfies the local flow-balance equations;
    a Rasengan transition replaces one closed cycle with the other.
    """
    if not cycle_lengths:
        raise ValueError("cycle_lengths cannot be empty")
    if any(length < 3 for length in cycle_lengths):
        raise ValueError("all cycle lengths must be at least 3")
    variable_count = 2 * sum(cycle_lengths)
    constraints: list[NDArray[np.float64]] = []
    rhs: list[float] = []
    feasible_state = np.zeros(variable_count, dtype=int)
    transitions = []
    coefficients = np.zeros(variable_count, dtype=float)
    variable_names: list[str] = []
    offset = 0
    for block, length in enumerate(cycle_lengths):
        first_cycle = list(range(offset, offset + length))
        second_cycle = list(range(offset + length, offset + 2 * length))
        for label, cycle in (("a", first_cycle), ("b", second_cycle)):
            for local in range(length - 1):
                row = np.zeros(variable_count, dtype=float)
                row[cycle[local]] = 1.0
                row[cycle[local + 1]] = -1.0
                constraints.append(row)
                rhs.append(0.0)
            variable_names.extend(f"cycle_{block}_{label}_{local}" for local in range(length))
        choice_row = np.zeros(variable_count, dtype=float)
        choice_row[first_cycle[0]] = 1.0
        choice_row[second_cycle[0]] = 1.0
        constraints.append(choice_row)
        rhs.append(1.0)
        feasible_state[first_cycle] = 1
        move = np.zeros(variable_count, dtype=int)
        move[first_cycle] = -1
        move[second_cycle] = 1
        transitions.append(move)
        for local, variable in enumerate(first_cycle):
            coefficients[variable] = 1.6 + 0.07 * block + 0.03 * local
        for local, variable in enumerate(second_cycle):
            coefficients[variable] = 1.2 + 0.05 * block + 0.02 * local
        offset += 2 * length
    problem = LinearConstraintBinaryProblem(
        linear=coefficients,
        constraints=np.asarray(constraints, dtype=float),
        rhs=np.asarray(rhs, dtype=float),
        sense="max",
        quadratic=_chain_quadratic(variable_count, -0.025),
        penalty=penalty,
        variable_names=tuple(variable_names),
    )
    return RasenganProblemInstance(
        problem=problem,
        transition_basis=np.asarray(transitions, dtype=int),
        feasible_state=feasible_state,
        search_space_size=2 ** len(cycle_lengths),
        feasible_states=tuple(
            tuple(state.tolist()) for state in _cycle_choice_feasible_states(cycle_lengths)
        ),
    )


def payment_settlement_instance(
    *,
    pair_count: int,
    penalty: float = 400.0,
) -> RasenganProblemInstance:
    """Build a liquidity-neutral reciprocal-payment batch instance."""
    if pair_count < 2:
        raise ValueError("pair_count must be at least 2")
    variable_count = 2 * pair_count
    constraints = np.zeros((pair_count + 1, variable_count), dtype=float)
    variable_names: list[str] = []
    for pair in range(pair_count):
        amount = float(3 + pair)
        constraints[pair, 2 * pair] = amount
        constraints[pair, 2 * pair + 1] = -amount
        variable_names.extend((f"payment_{pair}_out", f"payment_{pair}_return"))
    constraints[-1, :] = 1.0
    rhs = np.zeros(pair_count + 1, dtype=float)
    rhs[-1] = 2.0
    coefficients = np.asarray(
        [1.0 + 0.2 * pair + 0.08 * local for pair in range(pair_count) for local in range(2)],
        dtype=float,
    )
    quadratic = _chain_quadratic(variable_count, 0.025)
    problem = LinearConstraintBinaryProblem(
        linear=coefficients,
        constraints=constraints,
        rhs=rhs,
        sense="min",
        quadratic=quadratic,
        penalty=penalty,
        variable_names=tuple(variable_names),
    )
    feasible_state = np.zeros(variable_count, dtype=int)
    feasible_state[:2] = 1
    transitions = []
    for pair in range(1, pair_count):
        move = np.zeros(variable_count, dtype=int)
        move[:2] = -1
        move[2 * pair : 2 * pair + 2] = 1
        transitions.append(move)
    return RasenganProblemInstance(
        problem=problem,
        transition_basis=np.asarray(transitions, dtype=int),
        feasible_state=feasible_state,
        search_space_size=pair_count,
        feasible_states=tuple(tuple(state.tolist()) for state in _payment_pair_feasible_states(pair_count)),
    )


def payment_settlement_block_instance(
    *,
    block_pair_options: tuple[int, ...],
    penalty: float = 400.0,
) -> RasenganProblemInstance:
    """Build mixed liquidity-neutral payment blocks.

    A block with value ``m`` contains ``m`` reciprocal payment pairs. The local
    constraints enforce that exactly one reciprocal pair is selected, so every
    block remains liquidity-neutral.
    """
    if not block_pair_options:
        raise ValueError("block_pair_options cannot be empty")
    if any(option_count < 2 for option_count in block_pair_options):
        raise ValueError("each payment block needs at least two pair options")
    variable_count = 2 * sum(block_pair_options)
    constraints: list[NDArray[np.float64]] = []
    rhs: list[float] = []
    feasible_state = np.zeros(variable_count, dtype=int)
    transitions = []
    coefficients = np.zeros(variable_count, dtype=float)
    variable_names: list[str] = []
    offset = 0
    search_space = 1
    for block, option_count in enumerate(block_pair_options):
        pair_starts = []
        for option in range(option_count):
            out_index = offset + 2 * option
            return_index = out_index + 1
            pair_starts.append(out_index)
            neutral_row = np.zeros(variable_count, dtype=float)
            neutral_row[out_index] = 1.0
            neutral_row[return_index] = -1.0
            constraints.append(neutral_row)
            rhs.append(0.0)
            coefficients[out_index] = 1.4 + 0.06 * block + 0.03 * option
            coefficients[return_index] = 1.35 + 0.05 * block + 0.02 * option
            variable_names.extend(
                (
                    f"payment_block_{block}_option_{option}_out",
                    f"payment_block_{block}_option_{option}_return",
                )
            )
        choice_row = np.zeros(variable_count, dtype=float)
        for pair_start in pair_starts:
            choice_row[pair_start] = 1.0
        constraints.append(choice_row)
        rhs.append(1.0)
        feasible_state[offset] = 1
        feasible_state[offset + 1] = 1
        for option in range(1, option_count):
            move = np.zeros(variable_count, dtype=int)
            move[offset] = -1
            move[offset + 1] = -1
            move[offset + 2 * option] = 1
            move[offset + 2 * option + 1] = 1
            transitions.append(move)
        search_space *= option_count
        offset += 2 * option_count
    problem = LinearConstraintBinaryProblem(
        linear=coefficients,
        constraints=np.asarray(constraints, dtype=float),
        rhs=np.asarray(rhs, dtype=float),
        sense="max",
        quadratic=_chain_quadratic(variable_count, 0.015),
        penalty=penalty,
        variable_names=tuple(variable_names),
    )
    return RasenganProblemInstance(
        problem=problem,
        transition_basis=np.asarray(transitions, dtype=int),
        feasible_state=feasible_state,
        search_space_size=search_space,
        feasible_states=tuple(
            tuple(state.tolist()) for state in _payment_block_feasible_states(block_pair_options)
        ),
    )


def loan_feature_instance(
    *,
    group_count: int,
    penalty: float = 400.0,
) -> RasenganProblemInstance:
    """Build grouped feature-selection constraints with nonuniform groups."""
    group_sizes = _loan_group_sizes(group_count)
    coefficients = []
    for group, size in enumerate(group_sizes):
        for local in range(size):
            coefficients.append(1.25 + 0.19 * group + 0.11 * local)
    return _grouped_one_hot_instance(
        group_sizes=group_sizes,
        coefficients=np.asarray(coefficients, dtype=float),
        sense="max",
        quadratic_scale=-0.02,
        penalty=penalty,
        variable_prefix="loan_feature",
    )


def loan_feature_group_instance(
    *,
    group_sizes: tuple[int, ...],
    penalty: float = 400.0,
) -> RasenganProblemInstance:
    """Build loan feature selection with an explicit group-size pattern."""
    if not group_sizes:
        raise ValueError("group_sizes cannot be empty")
    coefficients = []
    for group, size in enumerate(group_sizes):
        if size < 2:
            raise ValueError("loan feature groups must have at least two options")
        for local in range(size):
            coefficients.append(1.25 + 0.19 * group + 0.11 * local)
    return _grouped_one_hot_instance(
        group_sizes=group_sizes,
        coefficients=np.asarray(coefficients, dtype=float),
        sense="max",
        quadratic_scale=-0.02,
        penalty=penalty,
        variable_prefix="loan_feature",
    )


def index_tracking_instance(
    *,
    sector_count: int,
    penalty: float = 400.0,
) -> RasenganProblemInstance:
    """Build sector one-hot asset selection constraints for index tracking."""
    group_sizes = _index_group_sizes(sector_count)
    exposures = []
    costs = []
    for sector, size in enumerate(group_sizes):
        for local in range(size):
            exposures.append(
                [
                    0.35 + 0.03 * sector + 0.02 * local,
                    0.42 + 0.01 * sector - 0.015 * local,
                    0.25 + 0.02 * ((sector + local) % 3),
                ]
            )
            costs.append(0.04 + 0.01 * local)
    exposure_matrix = np.asarray(exposures, dtype=float)
    target = np.asarray([0.55 * sector_count, 0.45 * sector_count, 0.32 * sector_count])
    linear = np.asarray(costs, dtype=float) - 2.0 * exposure_matrix @ target
    quadratic = exposure_matrix @ exposure_matrix.T
    instance = _grouped_one_hot_instance(
        group_sizes=group_sizes,
        coefficients=linear,
        sense="min",
        quadratic_scale=0.0,
        penalty=penalty,
        variable_prefix="index_asset",
    )
    problem = LinearConstraintBinaryProblem(
        linear=linear,
        constraints=instance.problem.constraints,
        rhs=instance.problem.rhs,
        sense="min",
        quadratic=quadratic,
        penalty=penalty,
        variable_names=instance.problem.variable_names,
    )
    return RasenganProblemInstance(
        problem=problem,
        transition_basis=instance.transition_basis,
        feasible_state=instance.feasible_state,
        search_space_size=instance.search_space_size,
        feasible_states=instance.feasible_states,
    )


def index_tracking_group_instance(
    *,
    group_sizes: tuple[int, ...],
    penalty: float = 400.0,
) -> RasenganProblemInstance:
    """Build index-tracking asset selection with explicit sector sizes."""
    if not group_sizes:
        raise ValueError("group_sizes cannot be empty")
    exposure_count = 3
    exposures = []
    costs = []
    for sector, size in enumerate(group_sizes):
        if size < 2:
            raise ValueError("index-tracking sectors must have at least two options")
        for local in range(size):
            exposures.append(
                [
                    0.35 + 0.03 * sector + 0.02 * local,
                    0.42 + 0.01 * sector - 0.015 * local,
                    0.25 + 0.02 * ((sector + local) % exposure_count),
                ]
            )
            costs.append(0.04 + 0.01 * local)
    exposure_matrix = np.asarray(exposures, dtype=float)
    target = np.asarray(
        [0.55 * len(group_sizes), 0.45 * len(group_sizes), 0.32 * len(group_sizes)]
    )
    linear = np.asarray(costs, dtype=float) - 2.0 * exposure_matrix @ target
    grouped = _grouped_one_hot_instance(
        group_sizes=group_sizes,
        coefficients=linear,
        sense="min",
        quadratic_scale=0.0,
        penalty=penalty,
        variable_prefix="index_asset",
    )
    quadratic = _sparse_tracking_quadratic(group_sizes, exposure_matrix)
    problem = LinearConstraintBinaryProblem(
        linear=linear,
        constraints=grouped.problem.constraints,
        rhs=grouped.problem.rhs,
        sense="min",
        quadratic=quadratic,
        penalty=penalty,
        variable_names=grouped.problem.variable_names,
    )
    return RasenganProblemInstance(
        problem=problem,
        transition_basis=grouped.transition_basis,
        feasible_state=grouped.feasible_state,
        search_space_size=grouped.search_space_size,
        feasible_states=grouped.feasible_states,
    )


def _grouped_one_hot_instance(
    *,
    group_sizes: tuple[int, ...],
    coefficients: NDArray[np.float64],
    sense: str,
    quadratic_scale: float,
    penalty: float,
    variable_prefix: str,
) -> RasenganProblemInstance:
    variable_count = sum(group_sizes)
    constraints = np.zeros((len(group_sizes), variable_count), dtype=float)
    variable_names: list[str] = []
    feasible_state = np.zeros(variable_count, dtype=int)
    transitions = []
    offset = 0
    for group, size in enumerate(group_sizes):
        constraints[group, offset : offset + size] = 1.0
        feasible_state[offset] = 1
        variable_names.extend(f"{variable_prefix}_{group}_{local}" for local in range(size))
        for local in range(1, size):
            move = np.zeros(variable_count, dtype=int)
            move[offset] = -1
            move[offset + local] = 1
            transitions.append(move)
        offset += size
    rhs = np.ones(len(group_sizes), dtype=float)
    problem = LinearConstraintBinaryProblem(
        linear=coefficients,
        constraints=constraints,
        rhs=rhs,
        sense=sense,  # type: ignore[arg-type]
        quadratic=_chain_quadratic(variable_count, quadratic_scale) if quadratic_scale else None,
        penalty=penalty,
        variable_names=tuple(variable_names),
    )
    return RasenganProblemInstance(
        problem=problem,
        transition_basis=np.asarray(transitions, dtype=int),
        feasible_state=feasible_state,
        search_space_size=int(np.prod(group_sizes)),
        feasible_states=tuple(
            tuple(state.tolist()) for state in enumerate_grouped_feasible_states(group_sizes)
        ),
    )


def _loan_group_sizes(group_count: int) -> tuple[int, ...]:
    if group_count <= 0:
        raise ValueError("group_count must be positive")
    pattern = (2, 3)
    return tuple(pattern[index % len(pattern)] for index in range(group_count))


def _index_group_sizes(sector_count: int) -> tuple[int, ...]:
    if sector_count <= 0:
        raise ValueError("sector_count must be positive")
    pattern = (3, 3, 4)
    return tuple(pattern[index % len(pattern)] for index in range(sector_count))


def _chain_quadratic(variable_count: int, scale: float) -> NDArray[np.float64]:
    quadratic = np.zeros((variable_count, variable_count), dtype=float)
    if scale == 0.0:
        return quadratic
    for index in range(variable_count - 1):
        quadratic[index, index + 1] = scale
        quadratic[index + 1, index] = scale
    return quadratic


def _sparse_tracking_quadratic(
    group_sizes: tuple[int, ...],
    exposure_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    variable_count = sum(group_sizes)
    quadratic = np.zeros((variable_count, variable_count), dtype=float)
    offsets = np.cumsum((0,) + group_sizes)
    for group in range(len(group_sizes) - 1):
        left_start = int(offsets[group])
        left_end = int(offsets[group + 1])
        right_start = int(offsets[group + 1])
        right_end = int(offsets[group + 2])
        width = min(left_end - left_start, right_end - right_start)
        for local in range(width):
            left = left_start + local
            right = right_start + local
            value = float(exposure_matrix[left] @ exposure_matrix[right])
            quadratic[left, right] = value
            quadratic[right, left] = value
    return quadratic


def enumerate_grouped_feasible_states(group_sizes: tuple[int, ...]) -> list[NDArray[np.int_]]:
    """Enumerate one-hot states for small grouped models."""
    states = []
    offsets = np.cumsum((0,) + group_sizes)
    for choices in product(*(range(size) for size in group_sizes)):
        state = np.zeros(int(offsets[-1]), dtype=int)
        for group, local in enumerate(choices):
            state[offsets[group] + local] = 1
        states.append(state)
    return states


def _cycle_feasible_states(cycle_count: int) -> list[NDArray[np.int_]]:
    states = []
    for selected_cycle in range(cycle_count):
        state = np.zeros(3 * cycle_count, dtype=int)
        state[3 * selected_cycle : 3 * selected_cycle + 3] = 1
        states.append(state)
    return states


def _cycle_choice_feasible_states(cycle_lengths: tuple[int, ...]) -> list[NDArray[np.int_]]:
    states = []
    variable_count = 2 * sum(cycle_lengths)
    offsets = np.cumsum((0,) + tuple(2 * length for length in cycle_lengths))
    for choices in product((0, 1), repeat=len(cycle_lengths)):
        state = np.zeros(variable_count, dtype=int)
        for block, choice in enumerate(choices):
            length = cycle_lengths[block]
            start = int(offsets[block]) + choice * length
            state[start : start + length] = 1
        states.append(state)
    return states


def _payment_pair_feasible_states(pair_count: int) -> list[NDArray[np.int_]]:
    states = []
    for selected_pair in range(pair_count):
        state = np.zeros(2 * pair_count, dtype=int)
        state[2 * selected_pair : 2 * selected_pair + 2] = 1
        states.append(state)
    return states


def _payment_block_feasible_states(
    block_pair_options: tuple[int, ...],
) -> list[NDArray[np.int_]]:
    states = []
    variable_count = 2 * sum(block_pair_options)
    offsets = np.cumsum((0,) + tuple(2 * option_count for option_count in block_pair_options))
    for choices in product(*(range(option_count) for option_count in block_pair_options)):
        state = np.zeros(variable_count, dtype=int)
        for block, option in enumerate(choices):
            start = int(offsets[block]) + 2 * option
            state[start : start + 2] = 1
        states.append(state)
    return states
