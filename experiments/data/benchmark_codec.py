"""Benchmark the optimized DataFrame codec against CSV."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from double_quant.data.codec import benchmark_dataframe_codec


def make_market_frame(rows: int, columns: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0004, scale=0.015, size=(rows, columns))
    prices = 100.0 * np.exp(returns.cumsum(axis=0))
    dates = pd.date_range("2020-01-02", periods=rows, freq="B", name="date")
    names = [f"asset_{index}" for index in range(columns)]
    return pd.DataFrame(prices, index=dates, columns=names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--columns", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument(
        "--compression",
        choices=["none", "deflate"],
        default="none",
        help="'none' favors speed; 'deflate' favors smaller payloads.",
    )
    parser.add_argument(
        "--assert-effect",
        action="store_true",
        help="Exit non-zero if the optimized codec is not smaller than CSV.",
    )
    args = parser.parse_args(argv)

    frame = make_market_frame(args.rows, args.columns, args.seed)
    result = benchmark_dataframe_codec(
        frame,
        repeats=args.repeats,
        compression=args.compression,
    )

    print("DataFrame codec benchmark")
    print(f"rows={args.rows} columns={args.columns} repeats={args.repeats}")
    print(f"compression={args.compression}")
    print(f"csv_bytes={result.csv_bytes}")
    print(f"optimized_bytes={result.optimized_bytes}")
    print(f"size_reduction={result.size_reduction_ratio:.2%}")
    print(f"csv_encode_ms={result.csv_encode_seconds * 1000:.3f}")
    print(f"optimized_encode_ms={result.optimized_encode_seconds * 1000:.3f}")
    print(f"encode_speedup={result.encode_speedup:.2f}x")
    print(f"csv_decode_ms={result.csv_decode_seconds * 1000:.3f}")
    print(f"optimized_decode_ms={result.optimized_decode_seconds * 1000:.3f}")
    print(f"decode_speedup={result.decode_speedup:.2f}x")

    if args.assert_effect and result.optimized_bytes >= result.csv_bytes:
        raise SystemExit("optimized codec did not reduce payload size")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
