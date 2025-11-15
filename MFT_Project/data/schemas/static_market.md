# Static Market Data Schema

Synthetic market data is stored as a pickled `pandas.DataFrame` with the
following properties:

- **Index**: `MultiIndex` with levels `(datetime, instrument)`.
- **Columns**: `MultiIndex` with top-level groups `feature` and `label`.
  - Feature columns follow the naming convention `$feat_<id>`.
  - Label columns follow the naming convention `LABEL<i>`.
- **Dtypes**: `float32` for both features and labels to match Qlib defaults.

The layout is intentionally compatible with `qlib.data.dataset.loader.StaticDataLoader`
so that the generated file can be consumed by `DataHandlerLP` without rewriting
loader logic.
