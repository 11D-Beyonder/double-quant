"""Double Quant — quantum computing for quantitative finance."""

from double_quant.algorithm.hhl import HHLSolver
from double_quant.algorithm.shapley import (
    BinaryEnumerationCalculator,
    PermutationEnumerationCalculator,
    PermutationMCCalculator,
    QAEOptions,
    QuantumShapleyCaculator,
    ShapleyCalculator,
)
from double_quant.application import RiskAttributor, PortfolioOptimizer
from double_quant.data.source import PriceSource, YFinanceSource
from double_quant.data.transform import (
    to_covariance,
    to_expected_returns,
    to_log_returns,
)

__all__ = [
    "PriceSource",
    "YFinanceSource",
    "to_log_returns",
    "to_covariance",
    "to_expected_returns",
    "HHLSolver",
    "ShapleyCalculator",
    "BinaryEnumerationCalculator",
    "PermutationEnumerationCalculator",
    "PermutationMCCalculator",
    "QuantumShapleyCaculator",
    "QAEOptions",
    "RiskAttributor",
    "PortfolioOptimizer",
]
