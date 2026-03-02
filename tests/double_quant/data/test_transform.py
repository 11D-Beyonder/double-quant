import numpy as np
import pandas as pd


def test_to_log_returns():
    from double_quant.data.transform import to_log_returns

    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [50.0, 55.0, 60.5]})
    returns = to_log_returns(prices)
    assert returns.shape == (2, 2)
    assert np.isclose(returns.iloc[0]["A"], np.log(110 / 100))


def test_to_covariance():
    from double_quant.data.transform import to_covariance

    np.random.seed(42)
    prices = pd.DataFrame(np.random.lognormal(size=(100, 3)).cumsum(axis=0), columns=["A", "B", "C"])
    cov = to_covariance(prices)
    assert cov.shape == (3, 3)
    assert np.allclose(cov, cov.T)  # symmetric


def test_to_expected_returns():
    from double_quant.data.transform import to_expected_returns

    np.random.seed(42)
    prices = pd.DataFrame(np.random.lognormal(size=(100, 2)).cumsum(axis=0), columns=["A", "B"])
    er = to_expected_returns(prices)
    assert er.shape == (2,)
