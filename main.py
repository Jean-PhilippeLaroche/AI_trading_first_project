import argparse
import logging
import numpy as np
import pandas as pd
import joblib
import atexit
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import torch.optim.lr_scheduler

from utils.data_utils import (
    prepare_data_for_ai,
    load_stock_csv,
    add_indicators,
    clean_data
)
from scripts.train import train_model, walk_forward_split, get_tensorboard_writer, launch_tensorboard
from scripts.evaluate import evaluate_model, visualize_model_performance
from scripts.backtest import run_backtest


# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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
        threshold=0.01,  # relative threshold for buy/sell (1% default)
        initial_balance=10000,
        transaction_cost=0.02,  # 2% total cost
        visualize=True
):
    logging.info(f"Starting pipeline for {ticker}")
    logging.info(f"Configuration: window={window_size}, epochs={epochs}, threshold={threshold * 100}%")

    # ---- Launch TensorBoard automatically ----
    tb_process = launch_tensorboard(logdir="runs", port=6006)

    # Register cleanup for TensorBoard
    if tb_process:
        def cleanup_tensorboard():
            tb_process.terminate()
            logging.info("TensorBoard stopped.")

        atexit.register(cleanup_tensorboard)

    # 1) Prepare scaled sequences (X,y) and scaler
    logging.info("Step 1: Preparing data...")
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
    logging.info("Step 2: Splitting data chronologically...")
    X_train, y_train, X_val, y_val = walk_forward_split(X, y, train_size=train_size)
    logging.info(f"Split: train={len(X_train)} samples, val={len(X_val)} samples")

    # 3) Start TensorBoard writer for this run
    logging.info("Step 3: Starting TensorBoard logging...")
    writer = get_tensorboard_writer()

    # 4) Train model
    logging.info(f"Step 4: Training model for {epochs} epochs...")
    model = train_model(
        X_train, y_train, X_val, y_val,
        input_size=X_train.shape[2],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        writer=writer,
        scaler=scaler
    )

    if model is None:
        logging.error("Model training failed. Exiting.")
        return

    # close writer
    try:
        writer.close()
    except Exception as e:
        logging.warning(f"Failed to close TensorBoard writer: {e}")

    # 5) Load cleaned dataframe for backtesting
    logging.info("Step 5: Loading data for backtesting...")
    df_raw = load_stock_csv(ticker)
    if df_raw is None:
        logging.error("Could not load raw CSV for ticker; exiting.")
        return

    df_ind = add_indicators(df_raw)
    df_clean = clean_data(df_ind)

    # Define feature columns (must match what was used in training)
    feature_columns = ["close", "RSI", "MACD", "MACD_Signal", "SMA"]
    feature_columns = [c for c in feature_columns if c in df_clean.columns]
    logging.info(f"Using feature columns: {feature_columns}")

    # 6) Run proper backtest on validation period
    logging.info("Step 6: Running backtest on validation data...")

    # Calculate validation period indices
    total_sequences = len(X)
    split_idx = int(total_sequences * train_size)

    # Validation starts at: window_size + split_idx (in original dataframe)
    val_start_idx = window_size + split_idx
    val_end_idx = len(df_clean)

    logging.info(f"Backtest period: index {val_start_idx} to {val_end_idx} ({val_end_idx - val_start_idx} days)")

    # Run backtest
    backtest_results = run_backtest(
        model=model,
        scaler=scaler,
        df=df_clean,
        feature_columns=feature_columns,
        window_size=window_size,
        initial_balance=initial_balance,
        transaction_cost_pct=transaction_cost,
        threshold=threshold,
        start_idx=val_start_idx,
        end_idx=val_end_idx
    )

    # 7) Extract results for visualization
    logging.info("Step 7: Preparing visualization data...")
    portfolio_history = backtest_results['portfolio_history']

    # Extract arrays for plotting
    val_dates = portfolio_history['date'].values
    actual_prices = portfolio_history['current_price'].values
    predicted_prices = portfolio_history['predicted_price'].values

    # Convert signals to -1/0/1 format for plotting
    signal_map = {'BUY': 1, 'HOLD': 0, 'SELL': -1}
    signals = portfolio_history['signal'].map(signal_map).values

    portfolio_values = portfolio_history['portfolio_value'].values

    # Get indicators for the validation period
    indicators = {}
    for ind in ["SMA", "RSI", "MACD", "MACD_Signal"]:
        if ind in df_clean.columns:
            indicators[ind] = df_clean[ind].iloc[val_start_idx:val_end_idx].values

    # 8) Save backtest results to file
    logging.info("Step 8: Saving backtest results...")
    results_summary = {
        'ticker': ticker,
        'window_size': window_size,
        'epochs': epochs,
        'threshold': threshold,
        'initial_balance': initial_balance,
        'transaction_cost': transaction_cost,
        'final_value': backtest_results['final_value'],
        'total_return': backtest_results['total_return'],
        'expected_return': backtest_results['expected_return'],
        'sharpe_ratio': backtest_results['sharpe_ratio'],
        'max_drawdown': backtest_results['max_drawdown'],
        'win_rate': backtest_results['win_rate'],
        'total_trades': backtest_results['total_trades'],
        'total_fees': backtest_results['total_fees'],
        'buy_hold_return': backtest_results['buy_hold_return'],
        'outperformance': backtest_results['outperformance']
    }

    # Save to JSON
    import json
    results_file = f'backtest_results_{ticker}.json'
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    logging.info(f"Results saved to {results_file}")

    # Save detailed trades to CSV
    if len(backtest_results['trades']) > 0:
        trades_file = f'backtest_trades_{ticker}.csv'
        backtest_results['trades'].to_csv(trades_file, index=False)
        logging.info(f"Trade history saved to {trades_file}")

    # 9) Visualize (calls plot_utils via evaluate.visualize_model_performance)
    if visualize:
        logging.info("Step 9: Generating visualizations...")
        try:
            visualize_model_performance(
                dates=val_dates,
                actual_prices=actual_prices,
                predicted_prices=predicted_prices,
                signals=signals,
                portfolio_values=portfolio_values,
                indicators=indicators if indicators else None
            )
        except Exception as e:
            logging.error(f"Visualization failed: {e}")
            logging.exception("Full traceback:")

    # 10) Print summary
    logging.info("\n" + "=" * 70)
    logging.info("PIPELINE COMPLETE - SUMMARY")
    logging.info("=" * 70)
    logging.info(f"Ticker: {ticker}")
    logging.info(f"Training epochs: {epochs}")
    logging.info(f"Validation period: {val_end_idx - val_start_idx} days")
    logging.info(f"-" * 70)
    logging.info(f"Initial Balance:        ${initial_balance:,.2f}")
    logging.info(f"Final Portfolio Value:  ${backtest_results['final_value']:,.2f}")
    logging.info(f"Total Return:           {backtest_results['total_return']:>6.2f}%")
    logging.info(f"Expected Return (daily):{backtest_results['expected_return']:>6.4f}%")
    logging.info(f"Buy & Hold Return:      {backtest_results['buy_hold_return']:>6.2f}%")
    logging.info(f"Outperformance:         {backtest_results['outperformance']:>6.2f}%")
    logging.info(f"-" * 70)
    logging.info(f"Sharpe Ratio:           {backtest_results['sharpe_ratio']:>6.3f}")
    logging.info(f"Max Drawdown:           {backtest_results['max_drawdown']:>6.2f}%")
    logging.info(f"Win Rate:               {backtest_results['win_rate']:>6.2f}%")
    logging.info(f"-" * 70)
    logging.info(f"Total Trades:           {backtest_results['total_trades']}")
    logging.info(f"Total Fees Paid:        ${backtest_results['total_fees']:,.2f}")
    logging.info(f"Transaction Cost:       {transaction_cost * 100}%")
    logging.info("=" * 70 + "\n")

    logging.info("Pipeline finished successfully!")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stock Trading AI - Train model and backtest strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data parameters
    parser.add_argument("--ticker", type=str, default="AAPL",
                        help="Stock ticker symbol")
    parser.add_argument("--window", type=int, default=60,
                        help="Window size for sequences")
    parser.add_argument("--train_size", type=float, default=0.8,
                        help="Fraction of data for training (rest for validation)")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")

    # Trading parameters
    parser.add_argument("--threshold", type=float, default=0.02,
                        help="Relative threshold for buy/sell signals (e.g., 0.02 = 2%%)")
    parser.add_argument("--balance", type=float, default=10000,
                        help="Initial balance for backtesting")
    parser.add_argument("--transaction_cost", type=float, default=0.02,
                        help="Transaction cost as fraction (e.g., 0.02 = 2%%)")

    # Visualization
    parser.add_argument("--no_viz", action="store_true",
                        help="Disable plotting at end")

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
        transaction_cost=args.transaction_cost,
        visualize=not args.no_viz
    )