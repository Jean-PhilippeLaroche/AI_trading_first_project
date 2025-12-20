import os
import argparse
import logging
import requests
import time
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Tuple

# Create logs directory if it doesn't exist
os.makedirs("../logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # minimum level to log
    format="%(asctime)s [%(levelname)s] %(message)s",  # how log lines look
    handlers=[
        logging.FileHandler("../logs/download_data.log"),  # write to file
        logging.StreamHandler()  # also show in console
    ]
)

POLYGON_API_KEY = "Set_custom_API_key"
if not POLYGON_API_KEY:
    logging.error("Environment variable POLYGON_API_KEY not set. Please set it before running.")

POLYGON_BASE = "https://api.polygon.io"


def parse_interval(interval: str) -> Tuple[int, str]:
    """
    Map intervals like '1d', '1m', '15m', '1h' to polygon range parameters:
    returns (multiplier, timespan)
    """
    interval = interval.lower().strip()
    if interval.endswith("m") and interval != "1m":
        # minute-based (e.g., 15m, 5m, 30m)
        num = int(interval[:-1])
        return num, "minute"
    if interval.endswith("m") and interval == "1m":
        return 1, "minute"
    if interval.endswith("h"):
        num = int(interval[:-1])
        return num, "hour"
    if interval.endswith("d"):
        num = int(interval[:-1])
        return num, "day"
    # fallback: try numeric minutes
    raise ValueError(f"Unsupported interval format: {interval}")


def _days_per_chunk_for_limit(multiplier: int, timespan: str, bar_limit: int = 50000) -> int:
    """
    Estimate a safe number of calendar days per chunk so that the expected number of bars
    returned does not exceed 'bar_limit'. Uses market hours (390 minutes / trading day).
    Adds safety factor.
    """
    if timespan == "minute":
        bars_per_trading_day = 390 / multiplier  # typical US trading day minutes (6.5h = 390m)
    elif timespan == "hour":
        bars_per_trading_day = 6.5 / multiplier
    elif timespan == "day":
        bars_per_trading_day = 1 / multiplier
    else:
        bars_per_trading_day = 1.0

    if bars_per_trading_day <= 0:
        bars_per_trading_day = 1.0

    # how many trading days we can request safely
    trading_days = max(1, int(bar_limit / bars_per_trading_day))
    # convert trading days estimate to calendar days by adding a small buffer:
    calendar_days = max(1, int(trading_days * 1.2))  # safety multiplier
    return calendar_days


def daterange_chunks(start_date: datetime, end_date: datetime, days_chunk: int) -> List[Tuple[datetime, datetime]]:
    """
    Yield (chunk_start, chunk_end) pairs between start_date and end_date inclusive,
    where each chunk spans at most days_chunk calendar days.
    """
    chunks = []
    curr = start_date
    while curr <= end_date:
        chunk_end = min(end_date, curr + timedelta(days=days_chunk - 1))
        chunks.append((curr, chunk_end))
        curr = chunk_end + timedelta(days=1)
    return chunks


def fetch_aggregates(ticker: str, multiplier: int, timespan: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    """
    Fetch aggregates from polygon for ticker between start_iso and end_iso (YYYY-MM-DD).
    This function automatically chunks the request if necessary.
    Returns a concatenated pandas DataFrame with timestamp index and columns:
    ['open','high','low','close','volume']
    """
    if not POLYGON_API_KEY:
        raise RuntimeError("POLYGON_API_KEY not set in environment.")

    start_dt = datetime.fromisoformat(start_iso)
    end_dt = datetime.fromisoformat(end_iso)
    if end_dt < start_dt:
        raise ValueError("end date must be >= start date")

    days_chunk = _days_per_chunk_for_limit(multiplier, timespan)
    logging.info(f"Using chunk size ~{days_chunk} calendar days for multiplier={multiplier} {timespan}")

    chunks = daterange_chunks(start_dt, end_dt, days_chunk)
    all_frames = []
    session = requests.Session()
    for i, (cstart, cend) in enumerate(chunks, start=1):
        from_str = cstart.strftime("%Y-%m-%d")
        to_str = cend.strftime("%Y-%m-%d")
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": POLYGON_API_KEY
        }
        url = f"{POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_str}/{to_str}"
        logging.info(f"Chunk {i}/{len(chunks)}: {from_str} -> {to_str}  URL: {url}")

        backoff = 1.0
        max_backoff = 60.0
        while True:
            try:
                resp = session.get(url, params=params, timeout=30)
            except requests.RequestException as e:
                logging.warning(
                    f"Network error fetching {ticker} chunk {from_str}->{to_str}: {e}. Backing off {backoff}s.")
                time.sleep(backoff)
                backoff = min(max_backoff, backoff * 2)
                continue

            if resp.status_code == 200:
                data = resp.json()
                # Polygon returns {"results": [...], "ticker": "...", ...}
                if "results" not in data or not data["results"]:
                    logging.info(f"No results for {ticker} in chunk {from_str}->{to_str}.")
                    break
                df = pd.DataFrame(data["results"])
                # polygon timestamps are in milliseconds since epoch in 't'
                if "t" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
                    df.set_index("timestamp", inplace=True)
                # rename columns
                rename_map = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
                df.rename(columns=rename_map, inplace=True)
                df = df[["open", "high", "low", "close", "volume"]]
                all_frames.append(df)
                # break out of backoff loop for this chunk
                break

            elif resp.status_code in (429, 503, 504):
                # rate limited or server busy; backoff and retry
                logging.warning(f"Rate limited or server busy (status {resp.status_code}). Backing off {backoff}s.")
                time.sleep(backoff)
                backoff = min(max_backoff, backoff * 2)
                continue
            elif resp.status_code == 404:
                logging.error(
                    f"Ticker {ticker} not found or endpoint 404 for chunk {from_str}->{to_str}. Response: {resp.text}")
                break
            else:
                logging.error(
                    f"Error fetching {ticker} chunk {from_str}->{to_str}: HTTP {resp.status_code} - {resp.text}")
                # don't retry forever; abort chunk
                break

        # polite short sleep between chunk requests
        time.sleep(0.25)

    if not all_frames:
        return pd.DataFrame()  # empty

    df_all = pd.concat(all_frames)
    # deduplicate index and sort
    df_all = df_all[~df_all.index.duplicated(keep="first")]
    df_all.sort_index(inplace=True)
    return df_all


def download_stock_data_polygon(ticker: str, start: str, end: str, interval: str, output_dir: str):
    """
    Download stock data from Polygon (aggregates) and save as CSV.
    """
    logging.info(f"Downloading {ticker} data from {start} to {end} at interval {interval}...")

    try:
        multiplier, timespan = parse_interval(interval)
    except ValueError as e:
        logging.error(f"Invalid interval: {e}")
        return None

    try:
        df = fetch_aggregates(ticker, multiplier, timespan, start, end)
        logging.info(f"Fetched {len(df)} rows for {ticker}.")
    except Exception as e:
        logging.error(f"Failed to fetch {ticker}: {e}")
        return None

    if df.empty:
        logging.warning(f"No data found for {ticker}.")
        return None

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save CSV
    output_path = os.path.join(output_dir, f"{ticker}.csv")
    # write index as timestamp column
    df.to_csv(output_path, index=True)
    logging.info(f"Data saved to {output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download stock data from Polygon (replaces yfinance version).")
    parser.add_argument("--tickers", type=str, nargs='+',
                        default=["AAPL", "MSFT", "AMZN", "JPM", "BAC", "XOM", "CAT", "WMT", "KO", "TSLA"],
                        help="Stock ticker symbols (space-separated for multiple)")
    parser.add_argument("--start", type=str, default="2018-09-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2023-09-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", type=str, default="15m", help="Data interval (e.g., 1d, 1h, 15m, 1m)")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Directory to save data")

    args = parser.parse_args()

    if not POLYGON_API_KEY:
        logging.error("POLYGON_API_KEY not found — aborting. Set POLYGON_API_KEY in your environment.")
        raise SystemExit(1)

    for ticker in args.tickers:
        output_path = os.path.join(args.output_dir, f"{ticker}.csv")

        # Caching check
        if os.path.exists(output_path):
            logging.info(f"File already exists for {ticker}, skipping download: {output_path}")
            continue

        logging.info(f"Starting download for {ticker} at {args.interval} interval...")
        try:
            df = download_stock_data_polygon(ticker, args.start, args.end, args.interval, args.output_dir)
            if df is None or df.empty:
                logging.warning(f"No data saved for {ticker}.")
                continue
        except Exception as e:
            logging.error(f"Failed to download {ticker}: {e}")
            continue