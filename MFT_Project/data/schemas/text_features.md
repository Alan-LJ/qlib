# Text Feature Schema

Synthetic textual features are persisted as a pickled `pandas.DataFrame` with:

- **Index**: `MultiIndex` `(datetime, instrument)` matching the market data.
- **Columns**: Single-level index where each column represents a dense
  embedding dimension, named `text_feat_<zero-padded-id>`.
- **Dtype**: `float32` for efficient interoperability with PyTorch tensors.

Downstream components are expected to align the text frame with market data
via the shared index before batching the tensors for model consumption.
