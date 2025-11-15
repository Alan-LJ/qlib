# Dataset Module

The dataset layer integrates Qlib's `DatasetH` abstraction with additional text
features. The key export is `MultiModalDataset`, which:

- Wraps a standard `DataHandlerLP` configuration for market data.
- Lazily loads static text embeddings aligned to the market index.
- Returns structured batches (`MultiModalBatch`) combining market and text
  views.

Developers can call `prepare_with_text` to obtain plain dictionaries compatible
with PyTorch data loaders or use the richer `MultiModalBatch` object when
features/labels need to be accessed separately.
