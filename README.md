![Double Quant logo](https://raw.githubusercontent.com/11D-Beyonder/double-quant/main/docs/assets/logo.png)

<div align="center">
<h1>
<em>Quant</em>um
<em>Quant</em>itative
</h1>
</div>
<br>

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![GitHub License](https://img.shields.io/github/license/11D-Beyonder/double-quant)](LICENSE)

# Double Quant

**Double Quant** is a Python library for quantitative finance workflows built around quantum-inspired and quantum-backed algorithms.

## Overview

Double Quant bridges quantum computing and quantitative finance with reusable data transforms, HHL-based solvers, Shapley-value algorithms, and higher-level portfolio and risk applications.

### Key Features

- Quantum-aware algorithms for linear solving and Shapley attribution
- Layered package design across data, common utilities, algorithms, and applications
- Full type hints for stronger IDE support and safer integrations
- Portfolio and risk workflows built on top of reusable primitives
- Experiment-friendly components that can be composed outside the core package

## Architecture

Double Quant is organized as a four-part package:

```text
double_quant.data
double_quant.common
double_quant.algorithm
double_quant.application
```

### Layer Responsibilities

- **Data layer** (`double_quant.data`): price sources and transforms from prices to returns, covariances, and expected returns
- **Common layer** (`double_quant.common`): shared models and metrics such as `LinearSystem` and risk helpers
- **Algorithm layer** (`double_quant.algorithm`): HHL solver variants plus classical and quantum Shapley calculators
- **Application layer** (`double_quant.application`): higher-level workflows such as portfolio optimization and risk attribution

## Installation

- Python 3.11 or higher

### Install from PyPI with uv

```bash
uv add double-quant
```

### Install with pip

```bash
pip install double-quant
```

### Install from source for development

```bash
git clone https://github.com/11D-Beyonder/double-quant.git
cd double-quant
uv sync
```

## Quick Start

### Example: Solving a Linear System

```python
import numpy as np

from double_quant import HHLSolver

A = np.array([[2.0, 1.0], [1.0, 2.0]])
b = np.array([1.0, 1.0])

solution = HHLSolver.solve(A, b)
print(solution)
```

### Example: Classical Risk Attribution

```python
import pandas as pd

from double_quant import BinaryEnumerationCalculator, RiskAttributor

returns = pd.DataFrame(
    {
        "AAPL": [0.01, -0.03, 0.02, 0.015],
        "MSFT": [0.008, -0.01, 0.018, 0.012],
        "TLT": [0.002, 0.001, -0.002, 0.003],
    }
)

src = RiskAttributor(
    returns,
    BinaryEnumerationCalculator,
    alpha=0.95,
    mode="es",
).attribute()

print(src)
```

## Core Components

### HHL Solver

`HHLSolver` is the main entry point for solving Hermitian linear systems with the project's HHL-based workflow.

### Shapley Calculators

Double Quant includes exact, Monte Carlo, and quantum Shapley-value calculators for attribution problems.

### Risk Attribution

`RiskAttributor` supports both direct expected-shortfall attribution (`mode="es"`) and the quantum-compatible risk-saving formulation (`mode="rs"`).

## Documentation

Project documentation lives in the [`docs/`](docs/) directory, including:

- [`docs/application/risk.md`](docs/application/risk.md)
- [`docs/solver/shapley.md`](docs/solver/shapley.md)
- [`docs/experiments/risk.md`](docs/experiments/risk.md)

## Development

### Running Tests

```bash
uv run pytest
```

### Building the Package

```bash
uv build
```

### Code Style

This project uses `ruff` for linting and `basedpyright` for type checking.

## Contributing

We welcome contributions. If you would like to contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat(scope): add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Commit Message Format

We follow the Angular commit message convention:

```text
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## License

This project is licensed under the GPLv3. See [LICENSE](LICENSE) for details.

## Acknowledgments

This project draws inspiration from:

- [Qiskit](https://github.com/Qiskit/qiskit)
- [Qiskit Finance](https://github.com/qiskit-community/qiskit-finance)
