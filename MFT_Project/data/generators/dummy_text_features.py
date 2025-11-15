"""Generate synthetic textual embeddings aligned with the market calendar.

Runs independently or as part of automation pipelines. Example usage:

    python -m MFT_Project.data.generators.dummy_text_features \
        --start 2023-01-02 --end 2023-03-31 --tickers 200 --dim 256

Outputs a pickled ``pandas.DataFrame`` with a ``MultiIndex`` (datetime,
 instrument) matching the market data, containing dense embedding vectors.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from MFT_Project.configs.constants import PATHS, WINDOW
from .static_market_data import generate_instruments

DEFAULT_DIMENSION = 256


@dataclass(frozen=True)
class TextFeatureConfig:
    """Parameters controlling dummy text feature generation."""

    start: str = WINDOW.train_start
    end: str = WINDOW.test_end
    ticker_count: int = 300
    embedding_dim: int = DEFAULT_DIMENSION
    seed: int = 11_2023
    output_root: Path = PATHS.data_dir
    filename: str = "dummy_text_features.pkl"
    calendar_filename: str = "calendars/day.txt"
    instrument_filename: str = "instruments/all.txt"

    @property
    def output_path(self) -> Path:
        return self.output_root / self.filename

    @property
    def calendar_path(self) -> Path:
        return self.output_root / self.calendar_filename

    @property
    def instrument_path(self) -> Path:
        return self.output_root / self.instrument_filename


def _load_calendar(path: Path) -> Iterable[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing calendar file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _load_instruments(path: Path) -> Iterable[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing instrument file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _resolve_calendar(config: TextFeatureConfig) -> Iterable[pd.Timestamp]:
    calendar_file = config.calendar_path
    if calendar_file.exists():
        values = _load_calendar(calendar_file)
        return pd.to_datetime(values)
    return pd.date_range(start=config.start, end=config.end, freq="B")


def _resolve_instruments(config: TextFeatureConfig) -> List[str]:
    instrument_file = config.instrument_path
    if instrument_file.exists():
        return list(_load_instruments(instrument_file))
    return generate_instruments(config.ticker_count)


def _build_index(days: Iterable[pd.Timestamp], instruments: Iterable[str]) -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [pd.Index(days, name="datetime"), pd.Index(list(instruments), name="instrument")]
    )


def _write_pickle(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(path)


def generate_text_features(config: TextFeatureConfig) -> None:
    rng = np.random.default_rng(config.seed)
    days = _resolve_calendar(config)
    instruments = _resolve_instruments(config)
    index = _build_index(days, instruments)
    data = rng.standard_normal((len(index), config.embedding_dim)).astype(np.float32)
    columns = [f"text_feat_{i:04d}" for i in range(config.embedding_dim)]
    frame = pd.DataFrame(data, index=index, columns=columns)
    _write_pickle(frame, config.output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic text feature embeddings.")
    parser.add_argument("--start", default=TextFeatureConfig.start, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=TextFeatureConfig.end, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--tickers", type=int, default=TextFeatureConfig.ticker_count, help="Number of synthetic instruments.")
    parser.add_argument("--dim", type=int, default=TextFeatureConfig.embedding_dim, help="Embedding dimension.")
    parser.add_argument("--seed", type=int, default=TextFeatureConfig.seed, help="Random seed for reproducibility.")
    parser.add_argument("--output-root", default=str(TextFeatureConfig.output_root), help="Directory for the pickle output.")
    parser.add_argument("--filename", default=TextFeatureConfig.filename, help="Pickle filename.")
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> TextFeatureConfig:
    return TextFeatureConfig(
        start=args.start,
        end=args.end,
        ticker_count=args.tickers,
        embedding_dim=args.dim,
        seed=args.seed,
        output_root=Path(args.output_root),
        filename=args.filename,
    )


def main() -> None:
    args = _parse_args()
    config = _build_config(args)
    generate_text_features(config)
    print(f"Synthetic text embeddings written to {config.output_path}")


if __name__ == "__main__":
    main()
