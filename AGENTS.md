# Double Quant Context

## Project Overview

**Double Quant** is a high-performance quantum computing framework designed for quantitative finance. It aims to bridge quantum algorithms (like HHL) with financial applications (portfolio optimization, pricing) using a layered architecture.

**Key Technologies:**
- **Language:** Python 3.11+
- **Quantum Stack:** Qiskit, Qiskit Aer
- **Math/Data:** Scipy, Matplotlib, Seaborn
- **Package Manager:** `uv`
- **Build System:** `hatchling`

## Architecture

The project follows a three-layer architecture:
1.  **Application Layer** (`src/double_quant/application`): High-level financial logic (e.g., Portfolio optimization).
2.  **Solver Layer** (`src/double_quant/solver`): Quantum algorithms (e.g., `QuantumLinearSolver`, `HHL`).
3.  **Data Layer** (`src/double_quant/common`, `src/double_quant/data`): Core data structures (e.g., `LinearSystem`) and utilities.

## Setup & Usage

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Install dependencies
uv sync
```

### Building

```bash
uv build
```

### Testing

Tests are located in `tests/` and use `pytest`.
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=double_quant
```

Never run test unless the user requests it voluntarily.

## Development Conventions

### Code Style

- **Type Hints:** The project uses full type hints (`py.typed` is present).
- **Linting:** Follows `ruff` guidelines (implied by documentation).
- **Directory Structure:** Uses the `src/` layout.

### Commit Messages

Follows the **Angular convention**:
```
<type>(<scope>): <description>

[optional body]
```
- **Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- **Example:** `feat(solver): add sapo optimizer integration`
