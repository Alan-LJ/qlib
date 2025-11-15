# Data Module

This directory will contain data access layers and reproducible scripts for
constructing the synthetic datasets required to test the MFT workflow. The
following layout is planned:

- `generators/` – scripts to build synthetic or live market/text features.
- `loaders/` – helpers to load static data into Qlib-compatible structures.
- `schemas/` – documentation of expected data formats.

Generated artefacts (e.g. `.pkl` files) are excluded from version control via
`.gitignore`. Only declarative scripts and schemas will live inside the repo.

## Available generators

- `static_market_data.py`: produces deterministic synthetic market bars for
testing.
- `dummy_text_features.py`: produces random embeddings aligned with the market
calendar.
- `tencent_market_data.py`: downloads historical bars from Tencent, writes
`static_market_data.pkl`, `calendars/day.txt`, and `instruments/all.txt`. Example:

	```powershell
	python -m MFT_Project.data.generators.tencent_market_data `
		--start 2024-01-01 --end 2024-06-30 `
		--tickers 000001.SZ,600519.SH
	```

		脚本默认绕过系统代理，避免命中本地代理返回的 502/403；若需要使用环境
		中的 `HTTP_PROXY`/`HTTPS_PROXY`，可追加 `--use-system-proxy`。

- `juhe_text_features.py`: fetches company news from Juhe, builds hashed
embeddings, and stores them as `juhe_text_features.pkl`. Provide your Juhe API
key via `--api-key` or the `JUHE_API_KEY` environment variable. Example:

	```powershell
	$env:JUHE_API_KEY = "your_key_here"
	python -m MFT_Project.data.generators.juhe_text_features `
		--start 2024-01-01 --end 2024-06-30 `
		--tickers 000001.SZ,600519.SH `
		--filename juhe_text_features.pkl
	```

		默认使用 `stock=szXXXXXX` 参数样式，并在必要时自动尝试携带端口 `:8080`
		的 Juhe 入口。如遇 `请求地址错误` 可显式设置
		`--endpoint http://web.juhe.cn:8080/finance/stock/news` 或改用
		`--param-style gid` 以适配其他版本的接口。若运行环境必须经过代理，追加
		`--use-system-proxy`。

Both generators produce artefacts compatible with the downstream loaders and
datasets used in the multi-modal training pipeline.
