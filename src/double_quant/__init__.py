"""Double Quant — quantum computing for quantitative finance."""

from double_quant.algorithm.hhl import HHLSolver
from double_quant.algorithm.qubo import (
    NumPyMinimumEigensolverSolver,
    QAOASolver,
    QUBOSolver,
    QUBOSolverResult,
    SamplingVQESolver,
)
from double_quant.algorithm.shapley import (
    BinaryEnumerationCalculator,
    PermutationEnumerationCalculator,
    PermutationMCCalculator,
    QAEOptions,
    QuantumShapleyCalculator,
    ShapleyCalculator,
)
from double_quant.application import RiskAttributor
from double_quant.common import IsingProblem, QUBOProblem
from double_quant.data.source import PriceSource, YFinanceSource
from double_quant.data.transform import (
    to_covariance,
    to_expected_returns,
    to_log_returns,
)
from double_quant.programming import (
    DecisionProblem,
    DecisionProgram,
    EuropeanCallPriceMeasure,
    ExpectedShortfallMeasure,
    FinancialProblem,
    FinancialProgram,
    MeasureFunction,
    ShapleyRiskContributionMeasure,
    ValuationProblem,
    ValuationProgram,
    dot,
    matmul,
    mean,
    quad_form,
    square,
    sum_,
)

__all__ = [
    "PriceSource",
    "YFinanceSource",
    "to_log_returns",
    "to_covariance",
    "to_expected_returns",
    "HHLSolver",
    "QUBOProblem",
    "IsingProblem",
    "ShapleyCalculator",
    "BinaryEnumerationCalculator",
    "PermutationEnumerationCalculator",
    "PermutationMCCalculator",
    "QuantumShapleyCalculator",
    "QAEOptions",
    "QUBOSolver",
    "QUBOSolverResult",
    "NumPyMinimumEigensolverSolver",
    "QAOASolver",
    "SamplingVQESolver",
    "RiskAttributor",
    "FinancialProgram",
    "FinancialProblem",
    "DecisionProgram",
    "DecisionProblem",
    "ValuationProgram",
    "ValuationProblem",
    "MeasureFunction",
    "ExpectedShortfallMeasure",
    "ShapleyRiskContributionMeasure",
    "EuropeanCallPriceMeasure",
    "sum_",
    "dot",
    "quad_form",
    "matmul",
    "mean",
    "square",
]
