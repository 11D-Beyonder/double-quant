from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

from double_quant.data.codec import (
    benchmark_dataframe_codec,
    decode_dataframe,
    encode_dataframe,
    read_dataframe,
    write_dataframe,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
DOC_DIR = REPO_ROOT / "tests" / "docs" / "76-data-codec-optimization"
IMAGE_DIR = DOC_DIR / "images"
BENCHMARK_IMAGE = IMAGE_DIR / "codec_benchmark_bar.png"


def _configure_chinese_font() -> None:
    preferred = [
        "Microsoft YaHei",
        "PingFang SC",
        "Hiragino Sans GB",
        "Hiragino Sans",
        "Heiti SC",
        "STHeiti",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def _market_frame(
    rows: int = 512,
    columns: int = 6,
    *,
    seed: int = 20260627,
    tz: str | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0004, scale=0.015, size=(rows, columns))
    prices = 100.0 * np.exp(returns.cumsum(axis=0))
    index = pd.date_range(
        "2020-01-02",
        periods=rows,
        freq="B",
        name="date",
        tz=tz,
    )
    names = [f"asset_{index}" for index in range(columns)]
    return pd.DataFrame(prices, index=index, columns=names)


def _print_benchmark_summary(prefix: str, benchmark) -> None:
    print(
        f"[{prefix}] 未优化字节={benchmark.csv_bytes}, "
        f"优化字节={benchmark.optimized_bytes}, "
        f"体积降低={benchmark.size_reduction_ratio:.2%}"
    )
    print(
        f"[{prefix}] 未优化编码={benchmark.csv_encode_seconds * 1000:.3f}ms, "
        f"优化编码={benchmark.optimized_encode_seconds * 1000:.3f}ms, "
        f"编码加速={benchmark.encode_speedup:.2f}x"
    )
    print(
        f"[{prefix}] 未优化解码={benchmark.csv_decode_seconds * 1000:.3f}ms, "
        f"优化解码={benchmark.optimized_decode_seconds * 1000:.3f}ms, "
        f"解码加速={benchmark.decode_speedup:.2f}x"
    )


def _write_benchmark_chart(benchmark) -> None:
    _configure_chinese_font()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("数据编解码优化效果对比", fontsize=14)

    panels = [
        (
            "编解码数据体积",
            "字节",
            [benchmark.csv_bytes, benchmark.optimized_bytes],
        ),
        (
            "编码耗时",
            "毫秒",
            [
                benchmark.csv_encode_seconds * 1000,
                benchmark.optimized_encode_seconds * 1000,
            ],
        ),
        (
            "解码耗时",
            "毫秒",
            [
                benchmark.csv_decode_seconds * 1000,
                benchmark.optimized_decode_seconds * 1000,
            ],
        ),
    ]

    for ax, (title, ylabel, values) in zip(axes, panels, strict=True):
        bars = ax.bar(["未优化编解码", "优化编解码"], values, color=["#7A869A", "#1F77B4"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}" if value < 1000 else f"{value:,.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(BENCHMARK_IMAGE, dpi=160)
    plt.close(fig)


def test_case_01_round_trip_market_price_frame():
    frame = _market_frame(rows=512, columns=6)
    decoded = decode_dataframe(encode_dataframe(frame))

    pd.testing.assert_frame_equal(decoded, frame, check_exact=True)
    print("[用例1] 股票价格矩阵二进制编解码后可精确还原。")


def test_case_02_round_trip_timezone_index_frame():
    frame = _market_frame(rows=128, columns=4, tz="Asia/Shanghai")
    decoded = decode_dataframe(encode_dataframe(frame))

    pd.testing.assert_frame_equal(decoded, frame, check_exact=True)
    print("[用例2] 带时区交易日期索引可正确编码和解码。")


def test_case_03_file_io_round_trip(tmp_path: Path):
    frame = _market_frame(rows=256, columns=5)
    payload_path = tmp_path / "market_frame.dqbin"

    write_dataframe(frame, payload_path)
    decoded = read_dataframe(payload_path)

    pd.testing.assert_frame_equal(decoded, frame, check_exact=True)
    assert payload_path.stat().st_size > 0
    print(f"[用例3] 文件写入与读取成功，编码数据体积={payload_path.stat().st_size}字节。")


def test_case_04_rejects_non_datetime_index():
    frame = pd.DataFrame({"asset_a": [100.0, 101.0, 102.0]})

    with pytest.raises(TypeError, match="DatetimeIndex"):
        encode_dataframe(frame)
    print("[用例4] 非DatetimeIndex输入被正确拒绝。")


def test_case_05_rejects_non_numeric_columns():
    frame = pd.DataFrame(
        {"asset_a": [100.0, 101.0], "ticker": ["AAPL", "MSFT"]},
        index=pd.date_range("2020-01-02", periods=2, freq="B", name="date"),
    )

    with pytest.raises(TypeError, match="numeric"):
        encode_dataframe(frame)
    print("[用例5] 非数值列输入被正确拒绝。")


def test_case_06_deflate_mode_reduces_binary_payload_size():
    frame = _market_frame(rows=2048, columns=8)
    uncompressed = encode_dataframe(frame, compression="none")
    compressed = encode_dataframe(frame, compression="deflate")

    assert len(compressed) < len(uncompressed)
    print(
        "[用例6] deflate模式进一步压缩二进制编码数据："
        f"{len(uncompressed)} -> {len(compressed)} 字节。"
    )


def test_case_07_benchmark_proves_optimization_effect():
    frame = _market_frame(rows=10000, columns=12)
    benchmark = benchmark_dataframe_codec(frame, repeats=5)
    _print_benchmark_summary("用例7", benchmark)
    _write_benchmark_chart(benchmark)

    assert benchmark.optimized_bytes < benchmark.csv_bytes
    assert benchmark.size_reduction_ratio >= 0.40
    assert benchmark.encode_speedup >= 5.0
    assert benchmark.decode_speedup >= 1.5
    assert BENCHMARK_IMAGE.exists()
    assert BENCHMARK_IMAGE.stat().st_size > 0
    print(f"[用例7] 柱状图已生成：{BENCHMARK_IMAGE.relative_to(REPO_ROOT)}")
