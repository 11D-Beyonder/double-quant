import numpy as np
import pandas as pd
from double_quant.common.metric import annualized_volatility, expected_shortfall
from double_quant.application.risk import RiskSavingValueFunction, RiskAttributor
from double_quant.solver.shapley import BinaryEnumerationCalculator


def test_metrics():
    # Create dummy returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 1000)

    vol = annualized_volatility(returns)
    es = expected_shortfall(returns, alpha=0.95)

    print(f"Annualized Volatility: {vol:.4f}")
    print(f"Expected Shortfall (95%): {es:.4f}")

    assert vol > 0
    assert es > 0


def test_risk_saving_model():
    # Create dummy returns for 3 assets
    np.random.seed(42)
    data = np.random.normal(0.001, 0.02, (1000, 3))
    df = pd.DataFrame(data, columns=["A", "B", "C"])

    rs_vfunc = RiskSavingValueFunction(df, alpha=0.95)

    # Test bitmask 1 (Asset A)
    # RS({A}) = ES({A}) - ES({A}) = 0
    assert np.isclose(rs_vfunc[1], 0.0)

    # Test bitmask 3 (Assets A and B)
    # RS({A, B}) = ES({A}) + ES({B}) - ES({A, B})
    # Due to diversification, ES({A, B}) <= ES({A}) + ES({B})
    # So RS({A, B}) should be >= 0
    rs_ab = rs_vfunc[3]
    print(f"RS(A, B): {rs_ab:.4f}")
    assert rs_ab >= -1e-10  # Floating point tolerance


def test_risk_attributor():
    # Create dummy returns for 3 assets
    np.random.seed(42)
    # Asset A: High risk
    # Asset B: Medium risk
    # Asset C: Low risk / Hedge (negatively correlated with A)
    returns_a = np.random.normal(0.001, 0.05, 1000)
    returns_b = np.random.normal(0.001, 0.02, 1000)
    returns_c = -0.5 * returns_a + np.random.normal(0, 0.01, 1000)

    df = pd.DataFrame({"High": returns_a, "Mid": returns_b, "Hedge": returns_c})

    # Use classical solver for verification
    attributor = RiskAttributor(df, BinaryEnumerationCalculator, alpha=0.95)
    results = attributor.attribute()

    print("Risk Attribution Results (SRC):")
    for asset, src in results.items():
        print(f"  {asset}: {src:.6f}")

    # Total SRC should equal Total ES
    total_es = expected_shortfall(df.mean(axis=1).values, alpha=0.95)
    total_src = sum(results.values())
    print(f"Total ES: {total_es:.6f}, Total SRC: {total_src:.6f}")
    assert np.isclose(total_es, total_src)

    # Hedge asset should likely have lower or even negative SRC
    assert results["Hedge"] < results["High"]
