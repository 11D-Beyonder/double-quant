from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from double_quant.common import IsingProblem, QUBOProblem


def qubo_to_ising(problem: QUBOProblem) -> IsingProblem:
    """Convert a QUBO problem into an equivalent Ising problem."""

    matrix = 0.5 * (problem.quadratic_matrix + problem.quadratic_matrix.T)
    diagonal = np.diag(matrix).copy()
    pair_terms = np.triu(matrix, k=1)

    linear_bias = -0.5 * diagonal - 0.5 * (pair_terms.sum(axis=0) + pair_terms.sum(axis=1))
    quadratic_matrix = 0.5 * pair_terms
    quadratic_matrix = quadratic_matrix + quadratic_matrix.T
    constant = (
        problem.constant
        + 0.5 * diagonal.sum()
        + 0.5 * pair_terms.sum()
    )

    return IsingProblem(
        linear_bias=linear_bias,
        quadratic_matrix=quadratic_matrix,
        constant=float(constant),
        variable_names=problem.variable_names,
    )


def ising_to_pauli_operator(problem: IsingProblem) -> SparsePauliOp:
    """Translate an Ising problem to a Z-basis Hamiltonian."""

    num_qubits = problem.num_variables
    identity = "I" * num_qubits
    terms: list[tuple[str, complex]] = [(identity, complex(problem.constant))]

    for index, coefficient in enumerate(problem.linear_bias):
        if coefficient == 0:
            continue
        label = _single_z_label(num_qubits, index)
        terms.append((label, complex(coefficient)))

    pair_terms = np.triu(problem.quadratic_matrix, k=1)
    rows, cols = np.nonzero(pair_terms)
    for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
        coefficient = pair_terms[row, col]
        label = _pair_z_label(num_qubits, row, col)
        terms.append((label, complex(coefficient)))

    return SparsePauliOp.from_list(terms)


def bits_to_spins(bits: np.ndarray | list[int]) -> np.ndarray:
    bit_array = np.asarray(bits, dtype=int)
    if not np.isin(bit_array, (0, 1)).all():
        raise ValueError("bits must contain only 0 or 1")
    return 1 - 2 * bit_array


def spins_to_bits(spins: np.ndarray | list[int]) -> np.ndarray:
    spin_array = np.asarray(spins, dtype=int)
    if not np.isin(spin_array, (-1, 1)).all():
        raise ValueError("spins must contain only -1 or 1")
    return ((1 - spin_array) // 2).astype(int)


def qiskit_bitstring_to_array(bitstring: str) -> np.ndarray:
    return np.fromiter((int(bit) for bit in reversed(bitstring)), dtype=int)


def array_to_bitstring(bits: np.ndarray | list[int]) -> str:
    bit_array = np.asarray(bits, dtype=int)
    if not np.isin(bit_array, (0, 1)).all():
        raise ValueError("bits must contain only 0 or 1")
    return "".join(str(int(bit)) for bit in bit_array.tolist())


def _single_z_label(num_qubits: int, index: int) -> str:
    label = ["I"] * num_qubits
    label[num_qubits - 1 - index] = "Z"
    return "".join(label)


def _pair_z_label(num_qubits: int, left: int, right: int) -> str:
    label = ["I"] * num_qubits
    label[num_qubits - 1 - left] = "Z"
    label[num_qubits - 1 - right] = "Z"
    return "".join(label)
