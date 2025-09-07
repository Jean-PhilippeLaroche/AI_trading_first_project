import os
import argparse
import yfinance as yf
import pandas as pd
import logging


# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # minimum level to log
    format="%(asctime)s [%(levelname)s] %(message)s",  # how log lines look
    handlers=[
        logging.FileHandler("logs/download_data.log"),  # write to file
        logging.StreamHandler()  # also show in console
    ]
)


# TODO: Move config (tickers, dates, output paths) to a YAML/JSON file if project grows


def download_stock_data(ticker, start, end, output_dir):
    """
    Download stock data from Yahoo Finance and save as CSV.

    Args:
        ticker (str): Stock ticker symbol (e.g., "AAPL").
        start (str): Start date in "YYYY-MM-DD".
        end (str): End date in "YYYY-MM-DD".
        output_dir (str): Directory to save the CSV file.
    """
    logging.info(f"Downloading {ticker} data from {start} to {end}...")

    # Fetch data with error handling
    try:
        df = yf.download(ticker, start=start, end=end)
        logging.info(f"Downloaded {len(df)} rows for {ticker}")
    except Exception as e:
        logging.error(f"Failed to download {ticker}: {e}")
        return None

    if df.empty:
        logging.warning(f"No data found for {ticker}.")
        return None

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save CSV
    output_path = os.path.join(output_dir, f"{ticker}.csv")
    df.to_csv(output_path)

    logging.info(f"Data saved to {output_path}")
    return df

    # TODO: Add caching check (skip download if file already exists)
    # TODO: Handle exceptions (network issues, invalid ticker, etc.)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download stock data from Yahoo Finance.")
    parser.add_argument("--tickers", type=str, nargs='+',
                        default=["AAPL", "MSFT", "AMZN", "JPM", "BAC", "XOM", "CAT", "WMT", "KO", "TSLA"],
                        help="Stock ticker symbols (space-separated for multiple)")
    parser.add_argument("--start", type=str, default="2018-09-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2023-09-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", type=str, default="15m", help="Data interval (e.g., 1d, 1h, 15m)")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Directory to save data")

    args = parser.parse_args()

    for ticker in args.tickers:
        output_path = os.path.join(args.output_dir, f"{ticker}.csv")

        # ----- Caching check -----
        if os.path.exists(output_path):
            logging.info(f"File already exists for {ticker}, skipping download: {output_path}")
            continue

        logging.info(f"Starting download for {ticker} at {args.interval} interval...")
        try:
            df = yf.download(ticker, start=args.start, end=args.end, interval=args.interval)
            logging.info(f"Downloaded {len(df)} rows for {ticker} ({args.interval})")
        except Exception as e:
            logging.error(f"Failed to download {ticker}: {e}")
            continue

        if df.empty:
            logging.warning(f"No data found for {ticker}. Skipping.")
            continue

        os.makedirs(args.output_dir, exist_ok=True)
        df.to_csv(output_path)
        logging.info(f"Data saved to {output_path}")