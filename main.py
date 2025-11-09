import argparse
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

# imports from your project
from utils.data_utils import (
    prepare_data_for_ai,
    load_stock_csv,
    add_indicators,
    clean_data
)
from scripts.train import train_model, walk_forward_split, get_tensorboard_writer, launch_tensorboard
from scripts.evaluate import evaluate_model, visualize_model_performance
from envs.trading_env import TradingEnv

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------
# Helper: inverse-scale predictions
# ---------------------------
def inverse_scale_close(scaled_values, scaler, feature_columns, target_col="close"):
    """
    Inverse-transform an array of scaled Close values using the MinMax scaler that was fit
    on the feature_columns. Returns values in original price units.

    scaled_values: 1D array (n,)
    scaler: fitted sklearn MinMaxScaler (fitted on df[feature_columns])
    feature_columns: list of columns the scaler was fit on (order must match)
    target_col: name of the target column (must be in feature_columns)
    """
    scaled_values = np.asarray(scaled_values).reshape(-1, 1)
    n = len(feature_columns)
    # Build all-zero matrix then fill the column for Close with scaled values
    arr = np.zeros((len(scaled_values), n), dtype=float)
    try:
        col_idx = feature_columns.index(target_col)
    except ValueError:
        raise ValueError(f"{target_col} not found in feature_columns: {feature_columns}")

    arr[:, col_idx] = scaled_values[:, 0]
    inv = scaler.inverse_transform(arr)
    return inv[:, col_idx]


# ---------------------------
# Main orchestration
# ---------------------------
def main(
    ticker="AAPL",
    window_size=20,
    train_size=0.8,
    epochs=10,
    batch_size=32,
    lr=1e-3,
    threshold=0.01,   # relative threshold for buy/sell (1% default)
    initial_balance=10000,
    visualize=True
):
    logging.info(f"Starting pipeline for {ticker}")

    # ---- Launch TensorBoard automatically ----
    tb_process = launch_tensorboard(logdir="runs", port=6006)

    # 1) Prepare scaled sequences (X,y) and scaler
    X, y, scaler = prepare_data_for_ai(
        ticker,
        data_dir=None,
        feature_columns=None,
        target_column="close",
        window_size=window_size
    )
    if X is None or y is None:
        logging.error("Data preparation failed. Exiting.")
        return

    logging.info(f"Prepared sequences: X={X.shape}, y={y.shape}")

    # 2) Walk-forward split
    X_train, y_train, X_val, y_val = walk_forward_split(X, y, train_size=train_size)
    logging.info(f"Split: train={len(X_train)} samples, val={len(X_val)} samples")

    # 3) Start TensorBoard writer for this run
    writer = get_tensorboard_writer()
    # (it is OK if you set writer=None to skip TB logging)

    # 4) Train model
    model = train_model(
        X_train, y_train, X_val, y_val,
        input_size=X_train.shape[2],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        writer=writer
    )
    if model is None:
        logging.error("Model training failed. Exiting.")
        return

    # close writer
    try:
        writer.close()
    except Exception as e:
        logging.warning(f"Failed to close TensorBoard writer: {e}")

    # 5) Evaluate on validation set (returns scaled predictions)
    val_loss, preds_scaled = evaluate_model(model, X_val, y_val, device="cpu")
    logging.info(f"Evaluation complete. Val loss: {val_loss:.6f}")

    # 6) Recreate feature_columns (must match prepare_data_for_ai internals)
    feature_columns = ["close", "RSI", "MACD", "MACD_Signal", "SMA"]
    # only keep those present in the original df (prepare_data_for_ai used same logic)
    # to get df columns we need to load the cleaned df
    df_raw = load_stock_csv(ticker)  # auto-locates data/raw if needed
    if df_raw is None:
        logging.error("Could not load raw CSV for ticker; skipping visualization.")
        return

    df_ind = add_indicators(df_raw)
    df_clean = clean_data(df_ind)
    feature_columns = [c for c in feature_columns if c in df_clean.columns]
    logging.info(f"Using feature columns for inverse-scaling: {feature_columns}")

    # 7) Inverse-scale predictions and validation targets back to price units
    preds_unscaled = inverse_scale_close(preds_scaled, scaler, feature_columns, target_col="close")
    y_val_unscaled = inverse_scale_close(y_val, scaler, feature_columns, target_col="close")

    # 8) Build mapping from sequence index -> original dataframe index
    # When create_sequences produced X,y it used indices: y corresponds to df index i+window_size for i in 0..len(df)-window_size-1
    seq_indices = list(range(window_size, len(df_clean)))  # length == number of sequences
    # Now split index for validation
    split_idx = int(len(X) * train_size)
    val_seq_indices = seq_indices[split_idx:]  # dataframe indices for each element in X_val / y_val

    # Sanity shapes
    assert len(val_seq_indices) == len(preds_unscaled) == len(y_val_unscaled)

    # 9) Build trading signals from predictions vs current price
    # current price for each validation step is df_clean['Close'].iloc[idx] (price at timestep)
    current_prices = df_clean['close'].iloc[val_seq_indices].values
    # decide: buy if predicted > current * (1 + threshold), sell if predicted < current * (1 - threshold)
    signals = np.zeros(len(preds_unscaled), dtype=int)  # -1 sell, 0 hold, 1 buy
    buy_mask = preds_unscaled > current_prices * (1.0 + threshold)
    sell_mask = preds_unscaled < current_prices * (1.0 - threshold)
    signals[buy_mask] = 1
    signals[sell_mask] = -1

    logging.info(f"Signals generated (val len={len(signals)}): buys={int(buy_mask.sum())}, sells={int(sell_mask.sum())}")

    # 10) Simulate actions in TradingEnv to obtain a proper portfolio curve
    # We'll step through the entire environment and apply our signals at the appropriate steps.
    env = TradingEnv(df_clean, initial_balance=initial_balance, window_size=window_size)
    obs, info = env.reset()

    # Convert val_seq_indices to a set for quick membership checks
    val_idx_to_pos = {idx: pos for pos, idx in enumerate(val_seq_indices)}

    # Step through environment: env.current_step starts at window_size after reset.
    done = False
    while not done:
        step_index = env.current_step  # this will be used for indexing
        # Decide action for this step: default hold (0)
        action = 0
        if step_index in val_idx_to_pos:
            pos = val_idx_to_pos[step_index]
            sig = signals[pos]
            if sig == 1:
                action = 1  # Buy
            elif sig == -1:
                action = 2  # Sell
            else:
                action = 0  # Hold
        # step env
        obs, reward, done, info = env.step(action)

    # Now env.history contains step indices and portfolio values/actions etc.
    # Build arrays aligned with val_seq_indices:
    # map step -> portfolio value
    step_to_portfolio = dict(zip(env.history['step'], env.history['portfolio_value']))
    portfolio_values_for_val = [step_to_portfolio.get(idx, None) for idx in val_seq_indices]
    # If any None (unlikely) replace with last known value
    last_val = initial_balance
    for i, v in enumerate(portfolio_values_for_val):
        if v is None:
            portfolio_values_for_val[i] = last_val
        else:
            last_val = v

    # 11) Indicators dictionary aligned to val indices (using df_clean)
    indicators = {}
    for ind in ["SMA", "RSI", "MACD", "MACD_Signal"]:
        if ind in df_clean.columns:
            indicators[ind] = df_clean[ind].iloc[val_seq_indices].values

    # 12) Prepare arrays/dates for visualization
    val_dates = df_clean.index[val_seq_indices]
    actual_prices_for_plot = np.array(current_prices)
    predicted_prices_for_plot = np.array(preds_unscaled)

    # 13) Visualize (calls plot_utils via evaluate.visualize_model_performance)
    if visualize:
        try:
            visualize_model_performance(
                dates=val_dates,
                actual_prices=actual_prices_for_plot,
                predicted_prices=predicted_prices_for_plot,
                signals=signals,
                portfolio_values=np.array(portfolio_values_for_val),
                indicators=indicators if indicators else None
            )
        except Exception as e:
            logging.error(f"Visualization failed: {e}")
            logging.exception("Full traceback:")

    logging.info("Main pipeline finished.")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main orchestration for training + evaluation")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--threshold", type=float, default=0.01, help="relative threshold for buy/sell")
    parser.add_argument("--balance", type=float, default=10000)
    parser.add_argument("--no_viz", action="store_true", help="Disable plotting at end")
    args = parser.parse_args()

    main(
        ticker=args.ticker,
        window_size=args.window,
        train_size=args.train_size,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        threshold=args.threshold,
        initial_balance=args.balance,
        visualize=not args.no_viz
    )