# MFT_Project

Multi-Modal Finance Toolkit (MFT) is a Qlib-integrated research project for
building explainable multi-modal quantitative trading models. The toolkit
focuses on combining structured market data with unstructured textual signals
and providing thorough diagnostics via SHAP explainability workflows.

This repository segment is organised as a standalone Python package under the
Qlib source tree. It contains configuration files, data preparation scripts,
model definitions, dataset wrappers, workflow utilities, and interpretability
assets required to run end-to-end experiments.

## High-level structure

- `configs/` – workflow configuration templates and shared runtime constants.
- `data/` – data access helpers and reproducible synthetic data generators.
- `dataset/` – Qlib dataset abstractions for multi-modal inputs.
- `model/` – PyTorch model definitions and training utilities.
- `explain/` – SHAP-based interpretability pipeline components.
- `workflow/` – workflow orchestration helpers for Qlib `qrun` jobs.
- `scripts/` – entry points for common project tasks (data generation,
  training, explainability reports, etc.).
- `utils/` – shared utilities (logging, time management, feature engineering
  helpers, etc.).

Each submodule will be populated incrementally as we work through the build
plan.

## Quick start

Run the end-to-end synthetic workflow from the repository root:

```bash
python -m MFT_Project.scripts.run_pipeline full-run \
  --start 2023-01-02 --end 2023-06-30 --tickers 200 --features 7 --text-dim 256
```

This command will generate synthetic market/text data and train the
`FusionModel` using the default workflow configuration. Predictions for the
`test` 分段会打印在控制台，也可以通过 `--output` 参数写入 CSV。
