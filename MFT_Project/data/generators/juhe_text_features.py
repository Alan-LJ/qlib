"""Fetch news data from Juhe and convert it into daily text embeddings."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Sequence

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from MFT_Project.configs.constants import PATHS
from MFT_Project.utils.tickers import normalise_tickers, to_tencent_symbol

JUHE_ENDPOINTS: Sequence[str] = (
    "https://web.juhe.cn/finance/stock/news",
    "http://web.juhe.cn/finance/stock/news",
    "http://web.juhe.cn:8080/finance/stock/news",
)
TOKEN_FILTER = str.maketrans({"\n": " ", "\r": " "})
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/118.0 Safari/537.36"
)


@dataclass(frozen=True)
class JuheNewsConfig:
    start: str
    end: str
    tickers: List[str]
    api_key: str
    embedding_dim: int = 256
    page_size: int = 40
    max_pages: int = 20
    sleep: float = 0.5
    endpoint: str = JUHE_ENDPOINTS[0]
    enable_port_fallback: bool = True
    use_system_proxy: bool = False
    param_style: str = "stock"
    output_root: Path = PATHS.data_dir
    filename: str = "juhe_text_features.pkl"
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

    def iter_endpoints(self) -> Sequence[str]:
        if not self.enable_port_fallback:
            return (self.endpoint,)
        dedup: List[str] = []
        for url in (self.endpoint, *JUHE_ENDPOINTS):
            if url not in dedup:
                dedup.append(url)
        return tuple(dedup)


class JuheDownloadError(RuntimeError):
    """Raised when Juhe data cannot be retrieved."""


def _read_ticker_file(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _resolve_tickers(args: argparse.Namespace) -> List[str]:
    raw: List[str] = []
    if args.ticker_file:
        raw.extend(_read_ticker_file(Path(args.ticker_file)))
    if args.tickers:
        for part in args.tickers.split(","):
            if part.strip():
                raw.append(part.strip())
    if not raw:
        raise ValueError("At least one ticker must be provided via --tickers or --ticker-file.")
    return normalise_tickers(raw)


def _resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key
    env_value = os.getenv("JUHE_API_KEY")
    if env_value:
        return env_value
    raise ValueError("Juhe API key must be provided via --api-key or JUHE_API_KEY environment variable.")


def _load_calendar(config: JuheNewsConfig) -> pd.DatetimeIndex:
    path = config.calendar_path
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            values = [line.strip() for line in handle if line.strip()]
        return pd.to_datetime(values)
    return pd.date_range(start=config.start, end=config.end, freq="B")


def _clean_text(text: str) -> str:
    return text.translate(TOKEN_FILTER).strip()


def _extract_timestamp(item: MutableMapping[str, str]) -> pd.Timestamp | None:
    for key in ("p", "time", "datetime", "pub_time", "publishtime", "ctime"):
        value = item.get(key)
        if value:
            try:
                return pd.to_datetime(value)
            except Exception:
                continue
    date_value = item.get("date")
    if date_value:
        try:
            return pd.to_datetime(date_value)
        except Exception:
            return None
    return None


def _extract_text(item: MutableMapping[str, str]) -> str:
    pieces: List[str] = []
    for key in ("title", "content", "digest", "summary", "brief"):
        value = item.get(key)
        if value:
            pieces.append(str(value))
    return _clean_text(" ".join(pieces))


def _tokenise(text: str) -> Iterable[str]:
    compact = "".join(ch for ch in text if not ch.isspace())
    length = len(compact)
    if length == 0:
        return []
    if length == 1:
        return [compact]
    return [compact[i : i + 2] for i in range(length - 1)]


def _vectorise(texts: Iterable[str], hashed_dim: int) -> tuple[np.ndarray, int]:
    vector = np.zeros(hashed_dim, dtype=np.float32)
    total_chars = 0
    for text in texts:
        cleaned = _clean_text(text)
        total_chars += len(cleaned)
        for token in _tokenise(cleaned):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
            index = int.from_bytes(digest, "little") % hashed_dim
            vector[index] += 1.0
    return vector, total_chars


def _fetch_news_for_symbol(session: requests.Session, ticker: str, config: JuheNewsConfig) -> Dict[pd.Timestamp, List[str]]:
    symbol = to_tencent_symbol(ticker)
    aggregated: Dict[pd.Timestamp, List[str]] = {}
    params_base = {
        "key": config.api_key,
        "pagesize": config.page_size,
    }
    if config.param_style == "stock":
        params_base.update({"stock": symbol, "type": "0"})
    else:
        params_base.update({"gid": symbol})
    start_ts = pd.Timestamp(config.start)
    end_ts = pd.Timestamp(config.end) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    for page in range(1, config.max_pages + 1):
        params = dict(params_base)
        params["page"] = page
        response = _issue_news_request(session, params=params, config=config)
        response.raise_for_status()
        payload = response.json()
        error_code = payload.get("error_code")
        if error_code != 0:
            raise JuheDownloadError(f"Juhe API returned error code {error_code}: {payload.get('reason')}")
        result = payload.get("result") or {}
        data = result.get("data") or result.get("list") or []
        if not data:
            break
        stop_paging = False
        for item in data:
            timestamp = _extract_timestamp(item)
            if timestamp is None:
                continue
            if timestamp < start_ts:
                stop_paging = True
                continue
            if timestamp > end_ts:
                continue
            text = _extract_text(item)
            if not text:
                continue
            day = timestamp.normalize()
            aggregated.setdefault(day, []).append(text)
        if stop_paging:
            break
        if config.sleep:
            time.sleep(config.sleep)
    return aggregated


def _build_session(config: JuheNewsConfig) -> requests.Session:
    session = requests.Session()
    session.trust_env = config.use_system_proxy
    retry = Retry(
        total=3,
        read=3,
        connect=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT, "Referer": "https://finance.juhe.cn/"})
    return session


def _issue_news_request(
    session: requests.Session,
    params: Dict[str, str],
    config: JuheNewsConfig,
) -> requests.Response:
    last_error: Exception | None = None
    for endpoint in config.iter_endpoints():
        try:
            response = session.get(endpoint, params=params, timeout=20)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("Unable to issue Juhe request; no endpoints attempted")


def generate_juhe_text_features(config: JuheNewsConfig) -> pd.DataFrame:
    if config.embedding_dim < 8:
        raise ValueError("embedding_dim should be at least 8 to accommodate hashed features.")
    hashed_dim = config.embedding_dim - 2
    calendar = _load_calendar(config).sort_values()
    instruments = config.tickers
    session = _build_session(config)
    news_map: Dict[str, Dict[pd.Timestamp, List[str]]] = {}
    for ticker in instruments:
        news_map[ticker] = _fetch_news_for_symbol(session, ticker, config)
        if config.sleep:
            time.sleep(config.sleep)
    columns = [f"text_feat_{i:04d}" for i in range(config.embedding_dim)]
    records: List[np.ndarray] = []
    index_tuples: List[tuple[pd.Timestamp, str]] = []
    for ticker in instruments:
        daily_news = news_map.get(ticker, {})
        for day in calendar:
            texts = daily_news.get(day, [])
            hashed_vector, total_chars = _vectorise(texts, hashed_dim)
            news_count = float(len(texts))
            avg_chars = float(total_chars / news_count) if news_count > 0 else 0.0
            combined = np.concatenate([hashed_vector, np.array([news_count, avg_chars], dtype=np.float32)])
            records.append(combined.astype(np.float32, copy=False))
            index_tuples.append((day, ticker))
    index = pd.MultiIndex.from_tuples(index_tuples, names=["datetime", "instrument"])
    frame = pd.DataFrame(records, index=index, columns=columns)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(config.output_path)
    return frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Juhe news and create hashed text embeddings.")
    parser.add_argument("--start", required=True, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--tickers", help="Comma-separated list of tickers matching market data.")
    parser.add_argument("--ticker-file", help="File containing tickers, one per line.")
    parser.add_argument("--api-key", help="Juhe API key (falls back to JUHE_API_KEY env var).")
    parser.add_argument("--embedding-dim", type=int, default=256, help="Total embedding dimension (>=8).")
    parser.add_argument("--page-size", type=int, default=40, help="Juhe API page size.")
    parser.add_argument("--max-pages", type=int, default=20, help="Maximum pages to fetch per ticker.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep duration between API calls.")
    parser.add_argument("--endpoint", default=JUHE_ENDPOINTS[0], help="Primary Juhe endpoint URL.")
    parser.add_argument(
        "--no-port-fallback",
        action="store_true",
        help="Disable automatic fallback to alternative Juhe endpoints (including :8080).",
    )
    parser.add_argument(
        "--use-system-proxy",
        action="store_true",
        help="Respect HTTP(S)_PROXY environment variables when fetching Juhe data.",
    )
    parser.add_argument(
        "--param-style",
        choices=["stock", "gid"],
        default="stock",
        help="Parameter naming used by the Juhe API variant (default uses stock=sz000001).",
    )
    parser.add_argument("--output-root", default=str(PATHS.data_dir), help="Directory for the pickle output.")
    parser.add_argument("--filename", default="juhe_text_features.pkl", help="Pickle filename.")
    parser.add_argument("--calendar-filename", default="calendars/day.txt", help="Calendar filename relative to the output root.")
    parser.add_argument("--instrument-filename", default="instruments/all.txt", help="Instrument filename relative to the output root.")
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> JuheNewsConfig:
    tickers = _resolve_tickers(args)
    api_key = _resolve_api_key(args)
    return JuheNewsConfig(
        start=args.start,
        end=args.end,
        tickers=tickers,
        api_key=api_key,
        embedding_dim=args.embedding_dim,
        page_size=args.page_size,
        max_pages=args.max_pages,
        sleep=args.sleep,
        endpoint=args.endpoint,
        enable_port_fallback=not args.no_port_fallback,
        use_system_proxy=args.use_system_proxy,
        param_style=args.param_style,
        output_root=Path(args.output_root),
        filename=args.filename,
        calendar_filename=args.calendar_filename,
        instrument_filename=args.instrument_filename,
    )


def main() -> None:
    args = _parse_args()
    config = _build_config(args)
    frame = generate_juhe_text_features(config)
    print(f"Generated text features with shape {frame.shape} at {config.output_path}.")


if __name__ == "__main__":
    main()
