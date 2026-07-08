from __future__ import annotations

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from double_quant.algorithm.shapley import BinaryEnumerationCalculator, ValueLoader
from double_quant.application.risk import RiskAttributor, RiskSavingValueFunction
from double_quant.common.metric import expected_shortfall
from double_quant.data.source import YFinanceSource
from double_quant.data.transform import (
    to_covariance,
    to_expected_returns,
    to_log_returns,
)


def _pipeline_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset_a": [50.0, 51.2, 50.8, 52.4, 53.1],
            "asset_b": [51.0, 52.2, 51.8, 53.4, 54.1],
            "asset_c": [52.0, 53.2, 52.8, 54.4, 55.1],
        },
        index=pd.to_datetime(
            ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"]
        ),
    )


def test_data_fetch_analysis_encoding_flow(monkeypatch):
    from double_quant.data import source as source_module

    prices = _pipeline_prices()

    def fake_yfinance_download(tickers, start, end, auto_adjust):
        assert tickers == ["asset_a", "asset_b", "asset_c"]
        assert start == "2020-01-01"
        assert end == "2020-01-09"
        columns = pd.MultiIndex.from_product([["Adj Close"], prices.columns])
        return pd.DataFrame(prices.to_numpy(), index=prices.index, columns=columns)

    monkeypatch.setattr(source_module.yf, "download", fake_yfinance_download)

    fetched_prices = YFinanceSource().fetch(
        ["asset_a", "asset_b", "asset_c"], "2020-01-01", "2020-01-09"
    )
    returns = to_log_returns(fetched_prices)
    covariance = to_covariance(fetched_prices)
    expected_returns = to_expected_returns(fetched_prices)
    attribution = RiskAttributor(
        returns,
        BinaryEnumerationCalculator,
        mode="es",
        alpha=0.75,
    ).attribute()

    assert fetched_prices.equals(prices)
    assert returns.shape == (4, 3)
    assert covariance.shape == (3, 3)
    assert np.allclose(covariance, covariance.T)
    assert expected_returns.shape == (3,)
    assert set(attribution) == {"asset_a", "asset_b", "asset_c"}

    print("[数据获取] 统一价格矩阵shape:", fetched_prices.shape)
    print("[分析] log收益率行列:", returns.shape)
    print("[分析] 协方差矩阵shape:", covariance.shape)
    print("[分析] 期望收益:", np.round(expected_returns, 6).tolist())
    print("[分析] Shapley ES风险归因:", {k: round(v, 8) for k, v in attribution.items()})

    assets = returns.columns.tolist()[:2]
    selected_mask = (1 << 0) | (1 << 1)
    decoded_assets = [asset for index, asset in enumerate(assets) if selected_mask & (1 << index)]
    pair_returns = returns[assets]
    risk_saving_value = RiskSavingValueFunction(pair_returns, alpha=0.75)[selected_mask]

    assert selected_mask == 3
    assert decoded_assets == assets
    assert risk_saving_value >= 0.0

    individual_es = np.array(
        [
            expected_shortfall(pair_returns[assets[0]].to_numpy(dtype=float), alpha=0.75),
            expected_shortfall(pair_returns[assets[1]].to_numpy(dtype=float), alpha=0.75),
        ],
        dtype=float,
    )
    max_es = float(individual_es.max())
    normalized_values = individual_es / max_es

    decoded_probabilities = []
    for control_state in range(2):
        circuit = QuantumCircuit(2)
        if control_state == 1:
            circuit.x(0)
        circuit.append(ValueLoader(normalized_values, num_control=1, normalization=False), [0, 1])
        decoded_probabilities.append(Statevector(circuit).probabilities([1])[1])

    restored_es = np.asarray(decoded_probabilities) * max_es

    assert np.allclose(decoded_probabilities, normalized_values)
    assert np.allclose(restored_es, individual_es)

    print("[编解码] 资产集合bitmask编码:", selected_mask)
    print("[编解码] bitmask解码资产:", decoded_assets)
    print("[编解码] 风险节省函数RS(S):", round(float(risk_saving_value), 8))
    print("[编解码] 原始ES:", np.round(individual_es, 8).tolist())
    print("[编解码] 量子振幅概率读回ES:", np.round(restored_es, 8).tolist())
    print("[验收结论] 已完成数据获取、分析、金融集合编码和量子振幅解码闭环")
