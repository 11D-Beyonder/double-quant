"""Optimized codecs for numeric market-data frames."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from os import PathLike
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


CompressionMode = Literal["none", "deflate"]

_FORMAT_NAME = "double_quant.dataframe"
_FORMAT_VERSION = 1


@dataclass(frozen=True)
class DataFrameCodecBenchmark:
    """Before/after measurements for the optimized DataFrame codec."""

    csv_bytes: int
    optimized_bytes: int
    csv_encode_seconds: float
    optimized_encode_seconds: float
    csv_decode_seconds: float
    optimized_decode_seconds: float

    @property
    def optimized_size_ratio(self) -> float:
        return self.optimized_bytes / self.csv_bytes if self.csv_bytes else 0.0

    @property
    def size_reduction_ratio(self) -> float:
        if not self.csv_bytes:
            return 0.0
        return (self.csv_bytes - self.optimized_bytes) / self.csv_bytes

    @property
    def encode_speedup(self) -> float:
        if self.optimized_encode_seconds == 0.0:
            return float("inf")
        return self.csv_encode_seconds / self.optimized_encode_seconds

    @property
    def decode_speedup(self) -> float:
        if self.optimized_decode_seconds == 0.0:
            return float("inf")
        return self.csv_decode_seconds / self.optimized_decode_seconds


def encode_dataframe(
    frame: pd.DataFrame,
    *,
    compression: CompressionMode = "none",
) -> bytes:
    """Encode a numeric DatetimeIndex DataFrame into a compact binary payload.

    The codec is intentionally scoped to the shape used by Double Quant data
    sources: numeric columns indexed by trading dates. Values are stored as a
    contiguous float64 matrix to avoid CSV text formatting and parsing overhead.
    """

    _validate_frame(frame)
    if compression not in ("none", "deflate"):
        raise ValueError("compression must be 'none' or 'deflate'")

    index = pd.DatetimeIndex(frame.index)
    values = np.ascontiguousarray(frame.to_numpy(dtype=np.float64, copy=False))
    metadata = {
        "format": _FORMAT_NAME,
        "version": _FORMAT_VERSION,
        "columns": [str(column) for column in frame.columns],
        "index_name": frame.index.name,
        "columns_name": frame.columns.name,
        "index_timezone": str(index.tz) if index.tz is not None else None,
        "index_unit": _datetime_unit(index),
        "index_freq": index.freqstr,
        "value_dtype": "float64",
    }
    metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")

    buffer = BytesIO()
    save = np.savez_compressed if compression == "deflate" else np.savez
    save(
        buffer,
        metadata=np.frombuffer(metadata_bytes, dtype=np.uint8),
        index_values=np.asarray(index.asi8, dtype=np.int64),
        values=values,
    )
    return buffer.getvalue()


def decode_dataframe(payload: bytes) -> pd.DataFrame:
    """Decode a payload produced by :func:`encode_dataframe`."""

    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
        _validate_metadata(metadata)
        values = np.asarray(archive["values"], dtype=np.float64)
        index_values = np.asarray(archive["index_values"], dtype=np.int64)

    timezone = metadata["index_timezone"]
    unit = metadata.get("index_unit", "ns")
    if timezone is None:
        index = pd.to_datetime(index_values, unit=unit)
    else:
        index = pd.to_datetime(index_values, unit=unit, utc=True).tz_convert(timezone)
    index.name = metadata["index_name"]
    if metadata.get("index_freq") is not None:
        try:
            index.freq = metadata["index_freq"]
        except ValueError:
            pass

    frame = pd.DataFrame(values, index=index, columns=metadata["columns"])
    frame.columns.name = metadata["columns_name"]
    return frame


def write_dataframe(
    frame: pd.DataFrame,
    path: str | PathLike[str],
    *,
    compression: CompressionMode = "none",
) -> None:
    """Encode and write a DataFrame payload to disk."""

    Path(path).write_bytes(encode_dataframe(frame, compression=compression))


def read_dataframe(path: str | PathLike[str]) -> pd.DataFrame:
    """Read and decode a DataFrame payload from disk."""

    return decode_dataframe(Path(path).read_bytes())


def benchmark_dataframe_codec(
    frame: pd.DataFrame,
    *,
    repeats: int = 5,
    compression: CompressionMode = "none",
) -> DataFrameCodecBenchmark:
    """Measure the optimized codec against an unoptimized text codec for the same frame."""

    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    csv_payload = _encode_csv(frame)
    optimized_payload = encode_dataframe(frame, compression=compression)

    csv_encode_seconds = _median_seconds(lambda: _encode_csv(frame), repeats)
    optimized_encode_seconds = _median_seconds(
        lambda: encode_dataframe(frame, compression=compression),
        repeats,
    )
    csv_decode_seconds = _median_seconds(lambda: _decode_csv(csv_payload), repeats)
    optimized_decode_seconds = _median_seconds(
        lambda: decode_dataframe(optimized_payload),
        repeats,
    )

    return DataFrameCodecBenchmark(
        csv_bytes=len(csv_payload),
        optimized_bytes=len(optimized_payload),
        csv_encode_seconds=csv_encode_seconds,
        optimized_encode_seconds=optimized_encode_seconds,
        csv_decode_seconds=csv_decode_seconds,
        optimized_decode_seconds=optimized_decode_seconds,
    )


def _validate_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("optimized DataFrame codec requires a DatetimeIndex")
    if frame.index.hasnans:
        raise ValueError("optimized DataFrame codec does not support NaT indexes")
    non_numeric = [
        str(column)
        for column, dtype in frame.dtypes.items()
        if not is_numeric_dtype(dtype)
    ]
    if non_numeric:
        columns = ", ".join(non_numeric)
        raise TypeError(f"optimized DataFrame codec requires numeric columns: {columns}")


def _validate_metadata(metadata: object) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("invalid DataFrame codec metadata")
    if metadata.get("format") != _FORMAT_NAME:
        raise ValueError("unsupported DataFrame codec format")
    if metadata.get("version") != _FORMAT_VERSION:
        raise ValueError("unsupported DataFrame codec version")
    if not isinstance(metadata.get("columns"), list):
        raise ValueError("invalid DataFrame codec column metadata")


def _datetime_unit(index: pd.DatetimeIndex) -> str:
    unit = getattr(index.dtype, "unit", None)
    if isinstance(unit, str):
        return unit
    dtype_text = str(index.dtype)
    start = dtype_text.find("[")
    end = dtype_text.find(",", start)
    if end == -1:
        end = dtype_text.find("]", start)
    if start != -1 and end != -1:
        return dtype_text[start + 1 : end]
    return "ns"


def _encode_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv().encode("utf-8")


def _decode_csv(payload: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(payload), index_col=0, parse_dates=True)


def _median_seconds(operation, repeats: int) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        operation()
        samples.append(perf_counter() - start)
    return median(samples)
