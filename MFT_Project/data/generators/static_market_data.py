"""Utilities for generating synthetic market data compatible with Qlib.

The script can be used as a CLI entry point:

    python -m MFT_Project.data.generators.static_market_data \
        --start 2023-01-02 --end 2023-03-31 --tickers 200

It produces three artefacts under ``MFT_Project/data`` by default:

- ``static_market_data.pkl``: MultiIndex pandas DataFrame with feature and
  label columns following Qlib conventions.
- ``calendars/day.txt``: Trading calendar listing.
- ``instruments/all.txt``: Universe definition for the generated instruments.

The generation process is deterministic given the ``--seed`` argument so that
unit tests and smoke checks can rely on fixed outputs.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from MFT_Project.configs.constants import PATHS, WINDOW

FEATURE_LEVEL = "feature"
LABEL_LEVEL = "label"
DEFAULT_LABEL = "LABEL0"


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration parameters controlling synthetic market data generation."""

    start: str = WINDOW.train_start
    end: str = WINDOW.test_end
    ticker_count: int = 300
    feature_count: int = 7
    seed: int = 7_2023
    output_root: Path = PATHS.data_dir
    calendar_filename: str = "calendars/day.txt"
    instrument_filename: str = "instruments/all.txt"
    market_filename: str = "static_market_data.pkl"

    @property
    def calendar_path(self) -> Path:
        return self.output_root / self.calendar_filename

    @property
    def instrument_path(self) -> Path:
        return self.output_root / self.instrument_filename

    @property
    def market_path(self) -> Path:
        return self.output_root / self.market_filename


def _build_calendar(start: str, end: str) -> pd.DatetimeIndex:
    trading_days = pd.date_range(start=start, end=end, freq="B")
    if trading_days.empty:
        raise ValueError("No business days found in the requested interval.")
    return trading_days


def generate_instruments(count: int) -> List[str]:
    if count <= 0:
        raise ValueError("Ticker count must be positive.")
    padding = int(math.log10(count)) + 3
    return [f"STK{i:0{padding}d}" for i in range(count)]


def _build_multiindex(days: Iterable[pd.Timestamp], instruments: Iterable[str]) -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [pd.Index(days, name="datetime"), pd.Index(list(instruments), name="instrument")]
    )


def _generate_feature_columns(feature_count: int) -> List[str]:
    if feature_count <= 0:
        raise ValueError("Feature count must be positive.")
    return [f"$feat_{i}" for i in range(feature_count)]


def _simulate_market_frame(index: pd.MultiIndex, feature_columns: List[str], rng: np.random.Generator) -> pd.DataFrame:
    feature_data = rng.standard_normal((len(index), len(feature_columns))).astype(np.float32)
    feature_frame = pd.DataFrame(feature_data, index=index, columns=pd.MultiIndex.from_product([[FEATURE_LEVEL], feature_columns]))

    label_values = rng.standard_normal(len(index)).astype(np.float32)
    label_frame = pd.DataFrame(label_values, index=index, columns=pd.MultiIndex.from_product([[LABEL_LEVEL], [DEFAULT_LABEL]]))

    combined = pd.concat([feature_frame, label_frame], axis=1)
    combined.sort_index(axis=1, inplace=True)
    return combined


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_calendar(days: Iterable[pd.Timestamp], path: Path) -> None:
    _ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for day in days:
            handle.write(f"{day.strftime('%Y-%m-%d')}\n")


def _write_instruments(instruments: Iterable[str], path: Path) -> None:
    _ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for ticker in instruments:
            handle.write(f"{ticker}\n")


def _write_market_pickle(frame: pd.DataFrame, path: Path) -> None:
    _ensure_parent_dir(path)
    frame.to_pickle(path)


def generate_market_data(config: GenerationConfig) -> None:
    """Generate synthetic market artefacts according to ``config``."""

    rng = np.random.default_rng(config.seed)
    calendar = _build_calendar(config.start, config.end)
    instruments = generate_instruments(config.ticker_count)
    index = _build_multiindex(calendar, instruments)
    feature_cols = _generate_feature_columns(config.feature_count)
    market_frame = _simulate_market_frame(index, feature_cols, rng)

    _write_calendar(calendar, config.calendar_path)
    _write_instruments(instruments, config.instrument_path)
    _write_market_pickle(market_frame, config.market_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic Qlib market data for the MFT project.")
    parser.add_argument("--start", default=GenerationConfig.start, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=GenerationConfig.end, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--tickers", type=int, default=GenerationConfig.ticker_count, help="Number of synthetic instruments to create.")
    parser.add_argument("--features", type=int, default=GenerationConfig.feature_count, help="Number of feature columns.")
    parser.add_argument("--seed", type=int, default=GenerationConfig.seed, help="Random seed for reproducibility.")
    parser.add_argument("--output-root", default=str(GenerationConfig.output_root), help="Base directory for artefacts.")
    return parser.parse_args()


def _build_config_from_args(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        start=args.start,
        end=args.end,
        ticker_count=args.tickers,
        feature_count=args.features,
        seed=args.seed,
        output_root=Path(args.output_root),
    )


def main() -> None:
    args = _parse_args()
    config = _build_config_from_args(args)
    generate_market_data(config)
    print(f"Synthetic market data written to {config.market_path}")


if __name__ == "__main__":
    main()
