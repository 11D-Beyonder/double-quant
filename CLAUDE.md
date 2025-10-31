# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Double Quant is a quantum quantitative finance project that aims to apply quantum computing to quantitative finance. The project is in early development stage with minimal implementation currently.

## Project Structure

- `src/double_quant/` - Main Python package directory
  - `__init__.py` - Contains a basic hello function
  - `py.typed` - Indicates this is a typed package
- `pyproject.toml` - Project configuration using modern Python packaging
- `README.md` - Basic project documentation

## Development Environment

- **Python Version**: Requires Python >=3.11
- **Package Manager**: Uses `uv` for dependency management and virtual environments
- **Build System**: Uses `hatchling` as the build backend
- **Project Type**: Modern Python package with pyproject.toml configuration

## Common Commands

### Installation
```bash
uv sync
```

### Build
```bash
uv build
```

### Testing
No tests are currently set up. When adding tests, they should follow standard Python testing conventions.

## Development Notes

- This is a new project with minimal codebase
- The project draws inspiration from Qiskit and Qiskit Finance
- Currently has no external dependencies beyond build tools
- Uses type hints (indicated by py.typed file)
- Follows modern Python packaging standards

## Future Architecture

Since this is a quantum quantitative finance project, the expected architecture will likely include:
- Quantum computing algorithms for financial modeling
- Integration with quantum computing frameworks (potentially Qiskit-based)
- Financial data processing and analysis modules
- Portfolio optimization and risk analysis tools

The codebase is currently in early development and will evolve as quantum finance algorithms are implemented.