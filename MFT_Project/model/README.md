# Model Module

The `model` package exposes PyTorch implementations used throughout the
multi-modal workflow. The primary export is `FusionModel`, a Qlib-compatible
estimator that:

- Encodes market features and text embeddings through separate MLP "towers".
- Concatenates the tower outputs and feeds them into a fusion head producing a
  scalar prediction.
- Supports early stopping on validation loss, configurable hidden dimensions,
  dropout, activation choice, and optional text branch.
- Provides `get_backbone()`/`get_device()` helpers so SHAP and other advanced
  tooling can operate directly on the underlying neural network.

Typical usage occurs through the workflow configuration, but the model can also
be instantiated programmatically for bespoke experiments.
