"""Download market data from Tencent and store it in the static format used by the project."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from MFT_Project.configs.constants import PATHS
from MFT_Project.utils.tickers import as_qlib_code, normalise_tickers, to_tencent_symbol

HTTPS_ENDPOINT = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
HTTP_ENDPOINT = "http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
DEFAULT_TIMEOUT = 10
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/118.0 Safari/537.36"
)


@dataclass(frozen=True)
class TencentMarketConfig:
    start: str
    end: str
    tickers: List[str]
    adjust: str = "qfq"
    frequency: str = "day"
    limit: int = 800
    sleep: float = 0.2
    timeout: int = DEFAULT_TIMEOUT
    http_fallback: bool = True
    use_system_proxy: bool = False
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


class TencentDownloadError(RuntimeError):
    """Raised when Tencent data cannot be retrieved or parsed."""


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


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fetch_symbol(session: requests.Session, symbol: str, config: TencentMarketConfig) -> pd.DataFrame:
    params = {
        "param": f"{symbol},{config.frequency},{config.start},{config.end},{config.limit},{config.adjust}",
    }
    response = _issue_request(session, params=params, config=config)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", {}).get(symbol)
    if not data:
        raise TencentDownloadError(f"No data returned for symbol {symbol}.")
    series_key_candidates = [f"{config.adjust}{config.frequency}", config.frequency, config.frequency.upper()]
    rows: Iterable[List[str]] = []
    for candidate in series_key_candidates:
        raw_rows = data.get(candidate)
        if raw_rows:
            rows = raw_rows
            break
    if not rows:
        raise TencentDownloadError(f"Missing kline rows for symbol {symbol}.")
    records: List[Dict[str, float]] = []
    for entry in rows:
        if len(entry) < 6:
            continue
        day = entry[0]
        record = {
            "datetime": pd.Timestamp(day),
            "open": _safe_float(entry[1]),
            "close": _safe_float(entry[2]),
            "high": _safe_float(entry[3]),
            "low": _safe_float(entry[4]),
            "volume": _safe_float(entry[5]),
            "amount": _safe_float(entry[6]) if len(entry) > 6 else float("nan"),
        }
        records.append(record)
    if not records:
        raise TencentDownloadError(f"No valid rows decoded for symbol {symbol}.")
    frame = pd.DataFrame.from_records(records).drop_duplicates(subset="datetime")
    frame.set_index("datetime", inplace=True)
    frame.sort_index(inplace=True)
    start_ts = pd.Timestamp(config.start)
    end_ts = pd.Timestamp(config.end)
    frame = frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)]
    if frame.empty:
        raise TencentDownloadError(f"Filtered data for symbol {symbol} is empty.")
    frame["pct_change"] = frame["close"].pct_change()
    frame["future_return"] = frame["close"].shift(-1) / frame["close"] - 1.0
    return frame


def _build_session(config: TencentMarketConfig) -> requests.Session:
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
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT, "Referer": "https://gu.qq.com/"})
    return session


def _issue_request(
    session: requests.Session,
    params: Dict[str, str],
    config: TencentMarketConfig,
) -> requests.Response:
    try:
        return session.get(HTTPS_ENDPOINT, params=params, timeout=config.timeout)
    except requests.exceptions.SSLError:
        if not config.http_fallback:
            raise
        return session.get(HTTP_ENDPOINT, params=params, timeout=config.timeout)


def _to_market_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    columns = [
        ("feature", "open"),
        ("feature", "high"),
        ("feature", "low"),
        ("feature", "close"),
        ("feature", "volume"),
        ("feature", "amount"),
        ("feature", "pct_change"),
        ("label", "LABEL0"),
    ]
    multi_columns = pd.MultiIndex.from_tuples(columns)
    target = pd.DataFrame(index=frame.index, columns=multi_columns, dtype=float)
    target[("feature", "open")] = frame["open"].astype("float32")
    target[("feature", "high")] = frame["high"].astype("float32")
    target[("feature", "low")] = frame["low"].astype("float32")
    target[("feature", "close")] = frame["close"].astype("float32")
    target[("feature", "volume")] = frame["volume"].astype("float32")
    target[("feature", "amount")] = frame["amount"].astype("float32")
    target[("feature", "pct_change")] = frame["pct_change"].astype("float32")
    target[("label", "LABEL0")] = frame["future_return"].astype("float32")
    instrument_index = pd.Index([ticker] * len(target), name="instrument")
    multi_index = pd.MultiIndex.from_arrays([target.index, instrument_index])
    target.index = multi_index
    return target


def _write_calendar(days: Iterable[pd.Timestamp], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for day in pd.Index(days).sort_values():
            handle.write(f"{day.strftime('%Y-%m-%d')}\n")


def _write_instruments(tickers: Iterable[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for ticker in sorted({as_qlib_code(item) for item in tickers}):
            handle.write(f"{ticker}\n")


def generate_tencent_market(config: TencentMarketConfig) -> pd.DataFrame:
    session = _build_session(config)
    frames: List[pd.DataFrame] = []
    for ticker in config.tickers:
        symbol = to_tencent_symbol(ticker)
        data = _fetch_symbol(session, symbol, config)
        frames.append(_to_market_frame(data, ticker))
        if config.sleep:
            time.sleep(config.sleep)
    if not frames:
        raise TencentDownloadError("No frames generated.")
    market = pd.concat(frames).sort_index()
    _write_calendar(market.index.get_level_values(0).unique(), config.calendar_path)
    _write_instruments(config.tickers, config.instrument_path)
    config.market_path.parent.mkdir(parents=True, exist_ok=True)
    market.to_pickle(config.market_path)
    return market


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Qlib-ready market data from Tencent.")
    parser.add_argument("--start", required=True, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--tickers", help="Comma-separated list of tickers (e.g. 000001.SZ,600519.SH).")
    parser.add_argument("--ticker-file", help="File containing tickers, one per line.")
    parser.add_argument("--adjust", default="qfq", help="Adjustment flag, e.g. qfq or hfq.")
    parser.add_argument("--frequency", default="day", help="Kline frequency, default is day.")
    parser.add_argument("--limit", type=int, default=800, help="Maximum number of records requested per call.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between requests to avoid rate limits.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--no-http-fallback",
        action="store_true",
        help="Disable automatic downgrade to HTTP when HTTPS handshake fails.",
    )
    parser.add_argument(
        "--use-system-proxy",
        action="store_true",
        help="Respect HTTP(S)_PROXY environment variables instead of bypassing them.",
    )
    parser.add_argument("--output-root", default=str(PATHS.data_dir), help="Directory for artefacts.")
    parser.add_argument("--calendar-filename", default="calendars/day.txt", help="Calendar filename relative to output root.")
    parser.add_argument("--instrument-filename", default="instruments/all.txt", help="Instrument filename relative to output root.")
    parser.add_argument("--market-filename", default="static_market_data.pkl", help="Pickle filename for market data.")
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> TencentMarketConfig:
    tickers = _resolve_tickers(args)
    return TencentMarketConfig(
        start=args.start,
        end=args.end,
        tickers=tickers,
        adjust=args.adjust,
        frequency=args.frequency,
        limit=args.limit,
        sleep=args.sleep,
        timeout=args.timeout,
        http_fallback=not args.no_http_fallback,
        use_system_proxy=args.use_system_proxy,
        output_root=Path(args.output_root),
        calendar_filename=args.calendar_filename,
        instrument_filename=args.instrument_filename,
        market_filename=args.market_filename,
    )


def main() -> None:
    args = _parse_args()
    config = _build_config(args)
    market = generate_tencent_market(config)
    print(f"Downloaded {len(market)} rows across {len(config.tickers)} tickers. Saved to {config.market_path}.")


if __name__ == "__main__":
    main()
