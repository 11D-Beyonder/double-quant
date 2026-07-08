import pandas as pd
import pytest


def _market_frame(rows: int = 1024, columns: int = 8) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(20260627)
    returns = rng.normal(loc=0.0004, scale=0.015, size=(rows, columns))
    prices = 100.0 * np.exp(returns.cumsum(axis=0))
    index = pd.date_range("2022-01-03", periods=rows, freq="B", name="date")
    names = [f"asset_{idx}" for idx in range(columns)]
    return pd.DataFrame(prices, index=index, columns=names)


def test_optimized_dataframe_codec_round_trips_market_data():
    from double_quant.data.codec import decode_dataframe, encode_dataframe

    frame = _market_frame()
    decoded = decode_dataframe(encode_dataframe(frame))

    pd.testing.assert_frame_equal(decoded, frame, check_exact=True)


def test_optimized_dataframe_codec_reduces_payload_size_against_csv():
    from double_quant.data.codec import benchmark_dataframe_codec

    frame = _market_frame(rows=2048, columns=10)
    benchmark = benchmark_dataframe_codec(frame, repeats=3)

    assert benchmark.optimized_bytes < benchmark.csv_bytes
    assert benchmark.size_reduction_ratio >= 0.35


def test_optimized_dataframe_codec_rejects_non_numeric_data():
    from double_quant.data.codec import encode_dataframe

    frame = pd.DataFrame(
        {"ticker": ["AAPL", "MSFT"]},
        index=pd.date_range("2022-01-03", periods=2, freq="B", name="date"),
    )

    with pytest.raises(TypeError, match="numeric"):
        encode_dataframe(frame)
