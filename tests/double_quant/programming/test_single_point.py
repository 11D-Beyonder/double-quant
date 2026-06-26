import numpy as np
import pandas as pd
import pytest

from double_quant.algorithm.hhl import HHLSolver
from double_quant.algorithm.qubo import NumPyMinimumEigensolverSolver
from double_quant.algorithm.shapley import BinaryEnumerationCalculator
from double_quant.application.risk import RiskAttributor
from double_quant.common.metric import expected_shortfall
from double_quant.programming import (
    DecisionProgram,
    ExpectedShortfallMeasure,
    ShapleyRiskContributionMeasure,
    ValuationProgram,
)


def test_decision_program_solves_qubo_end_to_end():
    program = DecisionProgram(
        name="single_point_qubo",
        kind="decision",
        domain="portfolio",
    )
    program.add_data("assets", ["asset_a", "asset_b"])
    program.add_output("selected_assets")
    x = program.add_variables("x", 2, vtype="binary")
    program.set_objective(
        -1.0 * x[0] - 2.0 * x[1] + 4.0 * x[0] * x[1],
        sense="minimize",
    )

    qubo = program.to_qubo_problem()
    result = NumPyMinimumEigensolverSolver().solve(qubo)

    assert qubo.variable_names == ["x_0", "x_1"]
    assert qubo.evaluate([0, 1]) == pytest.approx(-2.0)
    assert result.best_bitstring.tolist() == [0, 1]
    assert result.best_objective == pytest.approx(-2.0)


def test_decision_program_solves_hhl_linear_system_end_to_end():
    program = DecisionProgram(
        name="single_point_linear_system",
        kind="decision",
        domain="portfolio",
    )
    program.add_output("weights")
    x = program.add_variables("x", 2)
    program.add_constraints(
        [
            x[0] + x[1] == 3.0,
            x[0] - x[1] == 1.0,
        ]
    )

    system = program.to_linear_system()
    hhl_solution = HHLSolver.solve(
        system.matrix,
        system.vector,
        "sapo",
        max_qpe_qubits=4,
    )

    np.testing.assert_allclose(system.matrix, np.array([[1.0, 1.0], [1.0, -1.0]]))
    np.testing.assert_allclose(system.vector, np.array([3.0, 1.0]))
    np.testing.assert_allclose(hhl_solution, np.array([2.0, 1.0]), atol=1.0e-8)


def test_valuation_program_evaluates_expected_shortfall_end_to_end():
    returns = np.array([0.01, -0.03, 0.02, -0.08, -0.04, 0.03])
    program = ValuationProgram(
        name="single_point_expected_shortfall",
        kind="valuation",
        domain="risk",
    )
    program.add_data("portfolio_returns", returns)
    program.add_parameter("alpha", 0.75)
    program.set_measure(ExpectedShortfallMeasure, target="portfolio")
    program.add_output("expected_shortfall")

    assert program.evaluate() == pytest.approx(expected_shortfall(returns, 0.75))


def test_valuation_program_solves_shapley_risk_contribution_end_to_end():
    returns = pd.DataFrame(
        {
            "asset_a": [0.01, -0.02, 0.03, -0.04, 0.02, -0.01],
            "asset_b": [-0.01, 0.02, -0.03, 0.01, -0.05, 0.03],
        }
    )
    program = ValuationProgram(
        name="single_point_risk_contribution",
        kind="valuation",
        domain="risk_attribution",
    )
    program.add_data("asset_returns", returns)
    program.add_parameter("alpha", 0.75)
    program.add_parameter("mode", "es")
    program.add_parameter("solver_class", BinaryEnumerationCalculator)
    program.set_measure(
        ShapleyRiskContributionMeasure,
        target="portfolio",
        breakdown="asset",
    )
    program.add_output("risk_contribution")

    actual = program.evaluate()
    expected = RiskAttributor(
        returns,
        BinaryEnumerationCalculator,
        alpha=0.75,
        mode="es",
    ).attribute()

    assert actual == pytest.approx(expected)
