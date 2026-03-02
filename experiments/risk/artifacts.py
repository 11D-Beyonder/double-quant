from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

from double_quant.data.time_series import from_yfinance


QUANTUM_COMPARISON_FILES = [f"quantum_comparison_n{n}.csv" for n in (3, 4, 5, 6)]
EMPIRICAL_CASE_FILES = [
    "empirical_hidden_risk.csv",
]


REQUIRED_SNAPSHOT_FILES = (
    [
        "vol_buckets_metrics.csv",
        "vol_buckets_series.csv",
        "restoration_accuracy.csv",
        "equal_error_oracle_calls_summary.csv",
    ]
    + QUANTUM_COMPARISON_FILES
    + EMPIRICAL_CASE_FILES
)


@dataclass(frozen=True)
class ArtifactPaths:
    cache_dir: Path
    snapshot_dir: Path
    figure_dir: Path


def get_artifact_paths() -> ArtifactPaths:
    return ArtifactPaths(
        cache_dir=Path("experiments/risk/cache"),
        snapshot_dir=Path("docs/assets/risk/data"),
        figure_dir=Path("docs/assets/risk"),
    )


def require_snapshot_files(snapshot_dir: Path, file_names: list[str]) -> None:
    missing = [name for name in file_names if not (snapshot_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing snapshot files in {snapshot_dir}: {', '.join(sorted(missing))}"
        )


def write_manifest(
    output_dir: Path, params: dict[str, object], source_data: str
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "source_data": source_data,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    return manifest_path


class DataPreparation:
    def __init__(self, data_dir: str | Path = "experiments/risk/cache"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.data_dir / "experiment_data_clean.csv"

    def get_tickers(self) -> list[str]:
        high_vol = [
            "NVDA",
            "TSLA",
            "AMD",
            "MRNA",
            "PYPL",
            "ZM",
            "COIN",
            "RBLX",
            "NET",
            "CRWD",
            "DOCU",
            "PLTR",
            "NIO",
            "XPEV",
            "LI",
            "BIDU",
            "BABA",
            "JD",
            "PDD",
            "SE",
            "MELI",
            "ARKK",
            "ARKG",
            "TQQQ",
            "SOXL",
            "UDOW",
            "URTY",
            "UPRO",
            "ENPH",
            "SEDG",
            "SHOP",
            "TWLO",
            "ROKU",
            "U",
            "SNOW",
            "DDOG",
            "ZS",
            "OKTA",
        ]

        mid_vol = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "V",
            "MA",
            "JPM",
            "BAC",
            "WFC",
            "C",
            "GS",
            "MS",
            "BRK-B",
            "UNH",
            "JNJ",
            "PFE",
            "MRK",
            "ABBV",
            "LLY",
            "PG",
            "KO",
            "PEP",
            "COST",
            "WMT",
            "TGT",
            "HD",
            "LOW",
            "MCD",
            "SBUX",
            "NKE",
            "DIS",
            "CMCSA",
            "NFLX",
            "T",
            "VZ",
            "CVX",
            "XOM",
            "INTC",
            "CSCO",
            "ORCL",
            "IBM",
            "TXN",
            "QCOM",
            "ADBE",
            "CRM",
            "INTU",
            "NOW",
            "AMAT",
            "LRCX",
        ]

        low_vol = [
            "RSG",
            "WM",
            "ED",
            "DUK",
            "SO",
            "NEE",
            "AEP",
            "D",
            "PEG",
            "EXC",
            "AWK",
            "WEC",
            "ES",
            "ETR",
            "FE",
            "PPL",
            "CMS",
            "LNT",
            "ATO",
            "EVRG",
            "TLT",
            "IEF",
            "SHY",
            "GOVT",
            "SHV",
            "BIL",
            "LQD",
            "HYG",
            "JNK",
            "AGG",
            "GLD",
            "SLV",
            "DBC",
            "VIXY",
        ]

        return list(set(high_vol + mid_vol + low_vol))

    def download(
        self,
        start: str = "2020-04-01",
        end: str = "2022-04-01",
        use_cache: bool = True,
    ):
        cache_path = str(self.file_path) if use_cache else None
        return from_yfinance(self.get_tickers(), start, end, cache_path=cache_path)
