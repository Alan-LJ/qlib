"""Command-line utilities for running the MFT project workflow."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

import qlib
from qlib.utils import init_instance_by_config

from MFT_Project.configs.constants import PATHS
from MFT_Project.data import (
    GenerationConfig,
    TextFeatureConfig,
    generate_market_data,
    generate_text_features,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("mft_pipeline")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def cmd_generate_data(args: argparse.Namespace) -> None:
    market_cfg = GenerationConfig(
        start=args.start,
        end=args.end,
        ticker_count=args.tickers,
        feature_count=args.features,
        seed=args.market_seed,
        output_root=Path(args.output_root),
    )
    text_cfg = TextFeatureConfig(
        start=args.start,
        end=args.end,
        ticker_count=args.tickers,
        embedding_dim=args.text_dim,
        seed=args.text_seed,
        output_root=Path(args.output_root),
    )
    LOGGER.info("Generating market data -> %s", market_cfg.market_path)
    generate_market_data(market_cfg)
    LOGGER.info("Generating text features -> %s", text_cfg.output_path)
    generate_text_features(text_cfg)


def cmd_train(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)

    qlib_cfg = cfg.get("qlib_init", {})
    qlib.init(**qlib_cfg)
    LOGGER.info("Initialised Qlib with provider_uri=%s", qlib_cfg.get("provider_uri"))

    dataset_cfg = cfg["task"]["dataset"]
    model_cfg = cfg["task"]["model"]

    LOGGER.info("Building dataset: %s", dataset_cfg["class"])
    dataset = init_instance_by_config(dataset_cfg)
    LOGGER.info("Building model: %s", model_cfg["class"])
    model = init_instance_by_config(model_cfg)

    LOGGER.info("Starting training ...")
    model.fit(dataset)
    LOGGER.info("Training finished. Running inference on segment '%s' ...", args.segment)
    preds = model.predict(dataset, segment=args.segment)

    if args.output is not None:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(output_path)
        LOGGER.info("Predictions saved to %s", output_path)
    else:
        LOGGER.info("Sample predictions:\n%s", preds.head())


def cmd_full_run(args: argparse.Namespace) -> None:
    LOGGER.info("=== Step 1: Generating synthetic data ===")
    cmd_generate_data(args)
    LOGGER.info("=== Step 2: Training model ===")
    cmd_train(args)


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    default_data_root = str(PATHS.data_dir)
    parser.add_argument("--start", default=GenerationConfig.start, help="Inclusive start date")
    parser.add_argument("--end", default=GenerationConfig.end, help="Inclusive end date")
    parser.add_argument("--tickers", type=int, default=GenerationConfig.ticker_count, help="Number of instruments")
    parser.add_argument("--features", type=int, default=GenerationConfig.feature_count, help="Market feature dimension")
    parser.add_argument("--text-dim", type=int, default=TextFeatureConfig.embedding_dim, help="Text embedding dimension")
    parser.add_argument("--market-seed", type=int, default=GenerationConfig.seed, help="Random seed for market generator")
    parser.add_argument("--text-seed", type=int, default=TextFeatureConfig.seed, help="Random seed for text generator")
    parser.add_argument("--output-root", default=default_data_root, help="Output directory for generated artefacts")


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=str(PATHS.configs_dir / "workflow_config_multi_modal.yaml"),
        help="Workflow YAML path",
    )
    parser.add_argument("--segment", default="test", help="Segment to predict after training")
    parser.add_argument("--output", help="Optional CSV path to store predictions")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MFT Project pipeline helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_generate = subparsers.add_parser("generate-data", help="Generate synthetic market & text data")
    _add_generation_args(parser_generate)
    parser_generate.set_defaults(func=cmd_generate_data)

    parser_train = subparsers.add_parser("train", help="Train the fusion model using workflow config")
    _add_train_args(parser_train)
    parser_train.set_defaults(func=cmd_train)

    parser_full = subparsers.add_parser("full-run", help="Generate data then train in one go")
    _add_generation_args(parser_full)
    _add_train_args(parser_full)
    parser_full.set_defaults(func=cmd_full_run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
