import argparse
import logging
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_contour
)
import torch
import json
from datetime import datetime
from utils.data_utils import prepare_data_for_ai, load_stock_csv, add_indicators, clean_data
from scripts.train import train_model, walk_forward_split
from scripts.backtest import run_backtest

logging.basicConfig(level=logging.INFO)


def objective(trial, ticker, data_dir=None):
    """
    Objective function for Optuna to optimize.

    This function:
    1. Samples hyperparameters
    2. Trains model
    3. Runs backtest
    4. Returns performance metric (Sharpe ratio)

    Args:
        trial: Optuna trial object
        ticker: Stock ticker to train on
        data_dir: Optional data directory

    Returns:
        float: Sharpe ratio (or other metric to maximize)
    """

    # ============================================================
    # STEP 1: SAMPLE HYPERPARAMETERS
    # ============================================================

    # Model architecture hyperparameters
    window_size = trial.suggest_int('window_size', 30, 90, step=10)
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.4)

    # Training hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

    # Early stopping (to speed up bad configs)
    early_stopping_patience = trial.suggest_int('early_stopping_patience', 10, 20)

    # Trading hyperparameters
    threshold = trial.suggest_float('threshold', 0.005, 0.04)
    transaction_cost = trial.suggest_float('transaction_cost', 0.001, 0.01)

    # Log current trial
    logging.info(f"\n{'=' * 60}")
    logging.info(f"Trial {trial.number}: Testing configuration")
    logging.info(f"{'=' * 60}")
    logging.info(f"  window_size: {window_size}")
    logging.info(f"  hidden_size: {hidden_size}")
    logging.info(f"  num_layers: {num_layers}")
    logging.info(f"  dropout: {dropout:.3f}")
    logging.info(f"  learning_rate: {learning_rate:.6f}")
    logging.info(f"  batch_size: {batch_size}")
    logging.info(f"  threshold: {threshold:.3f}")
    logging.info(f"  transaction_cost: {transaction_cost:.3f}")

    try:
        # ============================================================
        # STEP 2: PREPARE DATA
        # ============================================================
        X, y, scaler = prepare_data_for_ai(
            ticker=ticker,
            data_dir=data_dir,
            feature_columns=None,
            target_column="close",
            window_size=window_size
        )

        if X is None or y is None or len(X) < 100:
            logging.warning(f"Trial {trial.number}: Data preparation failed")
            return 0.0  # Bad score for failed trial

        # Split data
        X_train, y_train, X_val, y_val = walk_forward_split(X, y, train_size=0.8)

        # ============================================================
        # STEP 3: TRAIN MODEL
        # ============================================================
        # Use fewer epochs during tuning (faster)
        # Early stopping will catch good models anyway
        model = train_model(
            X_train, y_train, X_val, y_val,
            input_size=X_train.shape[2],
            epochs=100,  # Max epochs (early stopping will cut short)
            batch_size=batch_size,
            lr=learning_rate,
            writer=None,  # No TensorBoard during tuning
            scaler=scaler,
            early_stopping_patience=early_stopping_patience,
            lr_scheduler_patience=max(3, early_stopping_patience // 2),
            lr_scheduler_factor=0.5
        )

        if model is None:
            logging.warning(f"Trial {trial.number}: Model training failed")
            return 0.0

        # ============================================================
        # STEP 4: BACKTEST
        # ============================================================
        # Load clean dataframe
        df_raw = load_stock_csv(ticker)
        if df_raw is None:
            logging.warning(f"Trial {trial.number}: Failed to load data for backtest")
            return 0.0

        df_ind = add_indicators(df_raw)
        df_clean = clean_data(df_ind)

        feature_columns = ["close", "RSI", "MACD", "MACD_Signal", "SMA"]
        feature_columns = [c for c in feature_columns if c in df_clean.columns]

        # Calculate validation period
        # Validation start and end
        val_start_idx = int(len(df_clean) * 0.8)
        val_end_idx = len(df_clean)

        # Run backtest
        results = run_backtest(
            model=model,
            scaler=scaler,
            df=df_clean,
            feature_columns=feature_columns,
            window_size=window_size,
            initial_balance=10000,
            transaction_cost_pct=transaction_cost,
            threshold=threshold,
            start_idx=val_start_idx,
            end_idx=val_end_idx
        )

        # ============================================================
        # STEP 5: RETURN PERFORMANCE METRIC
        # ============================================================
        # You can optimize different metrics:
        # - Sharpe ratio (risk-adjusted return) - RECOMMENDED
        # - Total return (raw profit)
        # - Outperformance vs buy-and-hold
        # - Win rate × total return (combined metric)

        sharpe_ratio = results['sharpe_ratio']
        total_return = results['total_return']
        outperformance = results['outperformance']

        logging.info(f"Trial {trial.number} Results:")
        logging.info(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
        logging.info(f"  Total Return: {total_return:.2f}%")
        logging.info(f"  Outperformance: {outperformance:.2f}%")

        # Report intermediate values for pruning
        trial.report(sharpe_ratio, step=0)

        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()

        # Return metric to MAXIMIZE
        return sharpe_ratio

    except Exception as e:
        logging.error(f"Trial {trial.number} failed with error: {e}")
        return 0.0  # Bad score for failed trial


def run_hyperparameter_tuning(
        ticker='AAPL',
        n_trials=50,
        timeout=None,
        data_dir=None,
        study_name=None
):
    """
    Run hyperparameter tuning using Bayesian optimization.

    Args:
        ticker: Stock ticker to optimize for
        n_trials: Number of configurations to try (default: 50)
        timeout: Maximum time in seconds (optional)
        data_dir: Data directory (optional)
        study_name: Name for study (optional, auto-generated)

    Returns:
        optuna.Study: Completed study with results
    """

    if study_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"{ticker}_tuning_{timestamp}"

    logging.info(f"Starting hyperparameter tuning for {ticker}")
    logging.info(f"Study name: {study_name}")
    logging.info(f"Number of trials: {n_trials}")
    if timeout:
        logging.info(f"Timeout: {timeout} seconds")

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',  # Maximize Sharpe ratio
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=0
        )
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, ticker, data_dir),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    # ============================================================
    # PRINT RESULTS
    # ============================================================
    logging.info(f"\n{'=' * 70}")
    logging.info("HYPERPARAMETER TUNING COMPLETE")
    logging.info(f"{'=' * 70}")
    logging.info(f"Best trial: {study.best_trial.number}")
    logging.info(f"Best Sharpe ratio: {study.best_value:.3f}")
    logging.info(f"\nBest hyperparameters:")
    for param, value in study.best_params.items():
        logging.info(f"  {param}: {value}")

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    results_file = f"tuning_results_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    results_data = {
        'study_name': study_name,
        'ticker': ticker,
        'n_trials': len(study.trials),
        'best_trial': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'all_trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': str(t.state)
            }
            for t in study.trials
        ]
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    logging.info(f"\nResults saved to: {results_file}")

    # ============================================================
    # GENERATE VISUALIZATIONS
    # ============================================================
    try:
        import matplotlib.pyplot as plt

        logging.info("\nGenerating visualizations...")

        # Optimization history
        fig1 = plot_optimization_history(study)
        fig1.write_html(f"optimization_history_{ticker}.html")
        logging.info(f"  Saved: optimization_history_{ticker}.html")

        # Parameter importances
        fig2 = plot_param_importances(study)
        fig2.write_html(f"param_importances_{ticker}.html")
        logging.info(f"  Saved: param_importances_{ticker}.html")

        # Contour plot (top 2 parameters)
        if len(study.best_params) >= 2:
            params = list(study.best_params.keys())[:2]
            fig3 = plot_contour(study, params=params)
            fig3.write_html(f"contour_{ticker}.html")
            logging.info(f"  Saved: contour_{ticker}.html")

        logging.info("\nOpen the HTML files in your browser to view interactive plots!")

    except Exception as e:
        logging.warning(f"Visualization generation failed: {e}")

    return study


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for stock trading AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--ticker", type=str, default="AAPL",
                        help="Stock ticker to optimize for")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of configurations to try")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Maximum time in seconds (optional)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory (optional)")

    args = parser.parse_args()

    # Run tuning
    study = run_hyperparameter_tuning(
        ticker=args.ticker,
        n_trials=args.n_trials,
        timeout=args.timeout,
        data_dir=args.data_dir
    )

    print("\n" + "=" * 70)
    print("BEST CONFIGURATION FOUND:")
    print("=" * 70)
    print(f"Sharpe Ratio: {study.best_value:.3f}")
    print("\nUse these parameters in your next training:")
    print("-" * 70)
    for param, value in study.best_params.items():
        print(f"--{param} {value}")
    print("=" * 70)