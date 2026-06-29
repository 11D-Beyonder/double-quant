from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def find_transition_basis(
    constraints: NDArray[np.float64],
    *,
    atol: float = 1e-9,
) -> NDArray[np.int_]:
    """Return a ``{-1,0,1}`` basis for feasible-state transitions.

    The vectors span directions ``d`` with ``A d = 0``. When a row-reduction
    vector has larger integer entries, it is omitted because the transition
    Hamiltonian driver used by Rasengan acts on ``{-1,0,1}`` flips.
    """
    matrix = np.asarray(constraints, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("constraints must be a two-dimensional array")
    if matrix.size == 0:
        return np.zeros((0, matrix.shape[1] if matrix.ndim == 2 else 0), dtype=int)

    echelon = _remove_zero_rows(_row_reduce(matrix, atol=atol), atol=atol)
    if echelon.shape[0] == 0:
        return np.eye(matrix.shape[1], dtype=int)

    pivot_columns, free_columns = _find_pivot_and_free_columns(echelon, atol=atol)
    if not free_columns:
        return np.zeros((0, matrix.shape[1]), dtype=int)

    pivot_matrix = echelon[:, pivot_columns]
    basis: list[NDArray[np.int_]] = []
    for free_column in free_columns:
        rhs = -echelon[:, free_column]
        pivot_values = np.linalg.solve(pivot_matrix, rhs)
        vector = np.zeros(matrix.shape[1], dtype=float)
        vector[free_column] = 1.0
        vector[pivot_columns] = pivot_values
        rounded = np.rint(vector).astype(int)
        if np.allclose(vector, rounded, atol=atol) and np.all(np.isin(rounded, [-1, 0, 1])):
            basis.append(rounded)

    if not basis:
        return np.zeros((0, matrix.shape[1]), dtype=int)
    return greedy_simplify_transition_basis(np.asarray(basis, dtype=int))


def greedy_simplify_transition_basis(
    transition_basis: NDArray[np.int_],
) -> NDArray[np.int_]:
    """Greedily reduce nonzero entries while staying in ``{-1,0,1}``."""
    basis = np.asarray(transition_basis, dtype=int).copy()
    for row_index in range(len(basis)):
        for other_index in range(row_index + 1, len(basis)):
            add_candidate = basis[row_index] + basis[other_index]
            sub_candidate = basis[row_index] - basis[other_index]
            current_nonzeros = np.count_nonzero(basis[row_index])
            if _is_valid_transition(add_candidate) and np.count_nonzero(add_candidate) < current_nonzeros:
                basis[row_index] = add_candidate
            elif _is_valid_transition(sub_candidate) and np.count_nonzero(sub_candidate) < current_nonzeros:
                basis[row_index] = sub_candidate
    return basis


def _row_reduce(
    matrix: NDArray[np.float64],
    *,
    atol: float,
) -> NDArray[np.float64]:
    reduced = np.asarray(matrix, dtype=float).copy()
    row_count, column_count = reduced.shape
    lead = 0
    for row in range(row_count):
        if lead >= column_count:
            break
        pivot = row
        while abs(reduced[pivot, lead]) <= atol:
            pivot += 1
            if pivot == row_count:
                pivot = row
                lead += 1
                if lead == column_count:
                    return reduced
        reduced[[pivot, row]] = reduced[[row, pivot]]
        divisor = reduced[row, lead]
        reduced[row] = reduced[row] / divisor
        for other_row in range(row_count):
            if other_row != row:
                factor = reduced[other_row, lead]
                reduced[other_row] = reduced[other_row] - factor * reduced[row]
        lead += 1
    reduced[np.abs(reduced) <= atol] = 0.0
    return reduced


def _remove_zero_rows(
    matrix: NDArray[np.float64],
    *,
    atol: float,
) -> NDArray[np.float64]:
    return matrix[np.any(np.abs(matrix) > atol, axis=1)]


def _find_pivot_and_free_columns(
    matrix: NDArray[np.float64],
    *,
    atol: float,
) -> tuple[list[int], list[int]]:
    pivot_columns: list[int] = []
    for row in matrix:
        nonzero_columns = np.nonzero(np.abs(row) > atol)[0]
        if len(nonzero_columns) > 0:
            pivot_columns.append(int(nonzero_columns[0]))
    free_columns = [index for index in range(matrix.shape[1]) if index not in pivot_columns]
    return pivot_columns, free_columns


def _is_valid_transition(vector: NDArray[np.int_]) -> bool:
    return bool(np.all(np.isin(vector, [-1, 0, 1])))
