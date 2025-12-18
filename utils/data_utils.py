import os
import pandas as pd
import logging
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# Block 1: Load stock data CSV
# -----------------------------
def load_stock_csv(ticker, data_dir=None):
    """
    Load stock CSV into a DataFrame, auto-locating 'data/raw' if not specified.

    Args:
        ticker (str): Stock ticker symbol.
        data_dir (str, optional): Directory where CSV files are stored.

    Returns:
        pd.DataFrame: DataFrame with stock data, indexed by date.
    """
    import os
    import logging
    import pandas as pd

    # Auto-locate data/raw
    if data_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        found = False
        # Walk up 3 levels to find project root
        for _ in range(3):
            candidate = os.path.join(base_dir, "data", "raw")
            if os.path.exists(candidate):
                data_dir = candidate
                found = True
                break
            base_dir = os.path.dirname(base_dir)
        if not found:
            logging.error("Could not find 'data/raw' folder in project tree.")
            return None

    file_path = os.path.join(data_dir, f"{ticker}.csv")

    if not os.path.exists(file_path):
        logging.error(f"CSV file not found for {ticker}: {file_path}")
        return None

    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    logging.info(f"Loaded {len(df)} rows for {ticker} from {file_path}")

    # Ensure columns we need exist
    expected_cols = ["open", "high", "low", "close", "volume"]
    for col in expected_cols:
        if col not in df.columns:
            logging.warning(f"Column {col} missing in {ticker} data")

    return df


def compute_rsi(df, period=14, column="close"):
    """
    Compute Relative Strength Index (RSI).

    Args:
        df (pd.DataFrame): Stock data.
        period (int): Lookback period for RSI.
        column (str): Column name to compute RSI on.

    Returns:
        pd.Series: RSI values.
    """

    # copying the dataframe so that it doesn't modify the original one
    df = df.copy()

    if column not in df.columns:
        logging.error(f"{column} not in DataFrame")
        return None

    # Ensure numeric
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[column])

    rsi = RSIIndicator(df[column], window=period).rsi()
    return rsi

# -----------------------------
# MACD
# -----------------------------
def compute_macd(df, column="close", fast=12, slow=26, signal=9):
    """
    Compute MACD and signal line.

    Args:
        df (pd.DataFrame): Stock data.
        column (str): Column name to compute MACD on.
        fast (int): Fast EMA period.
        slow (int): Slow EMA period.
        signal (int): Signal line EMA period.

    Returns:
        pd.DataFrame: Columns ['MACD', 'MACD_Signal']
    """

    # copying the dataframe so that it doesn't modify the original one
    df = df.copy()

    if column not in df.columns:
        logging.error(f"{column} not in DataFrame")
        return None

    # Ensure numeric
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[column])

    macd_indicator = MACD(df[column], window_slow=slow, window_fast=fast, window_sign=signal)
    macd_df = pd.DataFrame({
        "MACD": macd_indicator.macd(),
        "MACD_Signal": macd_indicator.macd_signal()
    }, index=df.index)
    return macd_df

# -----------------------------
# Moving Averages
# -----------------------------
def compute_moving_average(df, period=20, column="close"):
    """
    Compute Simple Moving Average (SMA).

    Args:
        df (pd.DataFrame): Stock data.
        period (int): Lookback period for SMA.
        column (str): Column name to compute SMA on.

    Returns:
        pd.Series: SMA values.
    """

    # copying the dataframe so that it doesn't modify the original one
    df = df.copy()

    if column not in df.columns:
        logging.error(f"{column} not in DataFrame")
        return None

    # Ensure numeric
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[column])

    sma = df[column].rolling(window=period).mean()
    return sma

def add_indicators(df, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9, sma_period=20, price_column="close"):
    """
    Add RSI, MACD, and SMA indicators to the DataFrame.

    Args:
        df (pd.DataFrame): Stock price data.
        rsi_period (int): Lookback period for RSI.
        macd_fast (int): Fast EMA period for MACD.
        macd_slow (int): Slow EMA period for MACD.
        macd_signal (int): Signal EMA period for MACD.
        sma_period (int): Lookback period for SMA.
        price_column (str): Column to use for indicators.

    Returns:
        pd.DataFrame: Original DataFrame with new columns:
                      ['RSI', 'MACD', 'MACD_Signal', 'SMA']
    """
    df = df.copy()

    # RSI
    df["RSI"] = compute_rsi(df, period=rsi_period, column=price_column)

    # MACD
    macd_df = compute_macd(df, column=price_column, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    if macd_df is not None:
        df = pd.concat([df, macd_df], axis=1)

    # SMA
    df["SMA"] = compute_moving_average(df, period=sma_period, column=price_column)

    logging.info(f"Indicators added: RSI({rsi_period}), MACD({macd_fast},{macd_slow},{macd_signal}), SMA({sma_period})")
    return df


def clean_data(df):
    """
    Clean the stock data DataFrame by handling missing values and ensuring numeric types.
    """
    df = df.copy()

    # Forward-fill missing data
    df = df.ffill()

    # Drop remaining NaNs if any
    df.dropna(inplace=True)

    # Ensure all columns are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    logging.info(f"Data cleaned: {len(df)} rows remain after cleaning")
    return df


def scale_features(df, feature_columns=None):
    """
    Scale the selected features using Min-Max scaling.
    """
    df = df.copy()
    if feature_columns is None:
        feature_columns = df.columns  # scale everything if not specified

    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    logging.info(f"Features scaled: {feature_columns}")
    return df, scaler


def create_sequences(df, feature_columns, target_column="close", window_size=20):
    """
    Convert DataFrame into sequences for AI training.

    Returns:
        X: np.array of shape (samples, window_size, features)
        y: np.array of shape (samples,)
    """
    data = df[feature_columns].values
    target = df[target_column].values

    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(data[i:i + window_size])
        y.append(target[i + window_size])

    X = np.array(X)
    y = np.array(y)

    logging.info(f"Created sequences: X={X.shape}, y={y.shape}")
    return X, y

def prepare_data_for_ai(
    ticker,
    data_dir=None,
    feature_columns=None,
    target_column="close",
    window_size=20,
    rsi_period=14,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    sma_period=20,
    start_idx=None,
    end_idx=None,
    scaler=None,
        df=None
):
    """
    Full pipeline to prepare stock data for AI training or validation.
    """

    # 1) Load full raw dataframe
    whole_df = load_stock_csv(ticker, data_dir)
    if whole_df is None:
        logging.error(f"CSV not found for {ticker}.")
        return None, None, None

    # 2) Compute indicators on FULL data (no leakage)
    whole_df = add_indicators(
        whole_df,
        rsi_period=rsi_period,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        sma_period=sma_period,
        price_column=target_column
    )

    whole_df = clean_data(whole_df)

    # 3) Slice AFTER indicator computation
    s = start_idx if start_idx is not None else 0
    e = end_idx if end_idx is not None else len(whole_df)
    df = whole_df.iloc[s:e].copy()

    # 4) Select features
    if feature_columns is None:
        feature_columns = ["close", "volume","RSI", "MACD", "MACD_Signal", "SMA"]
        feature_columns = [c for c in feature_columns if c in df.columns]

    # 5) Scale features
    if scaler is None:
        # Training phase: create and fit new scaler
        scaler = MinMaxScaler()
        scaler.fit(df[feature_columns])
        logging.info("Created and fitted new scaler")
    else:
        # Validation phase: use existing scaler (no fitting)
        logging.info("Using provided scaler (no refitting)")

    df_scaled = df.copy()
    df_scaled[feature_columns] = scaler.transform(df[feature_columns])

    # 6) Create sequences
    X, y = create_sequences(df_scaled, feature_columns,
                            target_column=target_column,
                            window_size=window_size)

    logging.info(f"Prepared {X.shape[0]} sequences for {ticker}")
    return X, y, scaler


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Example ticker
    ticker = "AAPL"

    # -----------------------------
    # Test 1: Load CSV
    # -----------------------------
    df = load_stock_csv(ticker)
    assert df is not None, "Failed to load CSV"
    assert len(df) > 0, "CSV loaded but empty"
    print("load_stock_csv passed")

    # -----------------------------
    # Test 2: Add indicators
    # -----------------------------
    df_ind = add_indicators(df)
    for col in ["RSI", "MACD", "MACD_Signal", "SMA"]:
        assert col in df_ind.columns, f"{col} not added"
    print("add_indicators passed")

    # -----------------------------
    # Test 3: Clean data
    # -----------------------------
    df_clean = clean_data(df_ind)
    assert df_clean.isnull().sum().sum() == 0, "Data still contains NaNs"
    print("clean_data passed")

    # -----------------------------
    # Test 4: Scale features
    # -----------------------------
    features = ["close", "volume", "RSI", "MACD", "MACD_Signal", "SMA"]
    df_scaled, scaler = scale_features(df_clean, feature_columns=features)
    assert np.all((df_scaled[features] >= 0) & (df_scaled[features] <= 1)), "Scaling failed"
    print("scale_features passed")

    # -----------------------------
    # Test 5: Create sequences
    # -----------------------------
    window_size = 20
    X, y = create_sequences(df_scaled, feature_columns=features, window_size=window_size)
    assert X.shape[0] == y.shape[0], "Mismatch between X and y"
    assert X.shape[1] == window_size, f"Sequence window size incorrect, expected {window_size}"
    assert X.shape[2] == len(features), f"Number of features incorrect, expected {len(features)}"
    print("create_sequences passed")

    # -----------------------------
    # Test 6: Full pipeline
    # -----------------------------
    X_full, y_full, scaler_full = prepare_data_for_ai(ticker)
    assert X_full.shape[0] > 0, "Pipeline returned empty X"
    assert y_full.shape[0] > 0, "Pipeline returned empty y"
    print("prepare_data_for_ai pipeline passed")

    print("\nAll tests passed successfully! 🎉")