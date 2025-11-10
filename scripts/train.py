"""
train.py
---------
This script trains a stock prediction AI using indicators (RSI, MACD, SMA, etc.).
The training process:
1. Load and prepare data (from data_utils).
2. Define a neural network model.
3. Train the model on historical sequences.
4. Save the trained model for later inference.

TODO:
- Add support for multiple tickers (portfolio-level training).
- Hyperparameter tuning (grid search, optuna, etc.).
- Support multiple walk-forward folds (not just one train/val split).
- Integrate wandb for richer experiment tracking if needed.
- Add more complex architectures: GRU, Transformers.
- Add multi-horizon forecasting (predict multiple days ahead).
- Add feature-wise attention / embedding layers for indicators.
- Support multi-ticker inputs as batch sequences.
- Support multiple tickers by iterating over a list of tickers
- Experiment with more advanced architectures (GRU, Transformer)
"""

# -----------------------------
# Imports
# -----------------------------
import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils.data_utils import prepare_data_for_ai
from torch.utils.tensorboard import SummaryWriter
import datetime
import subprocess
import webbrowser
import time
import platform
import uuid
import shutil
import joblib
from pathlib import Path

def launch_tensorboard(logdir="runs", port=6006):
    """
    Launch TensorBoard as a subprocess and open it in the browser.

    Args:
        logdir (str): Directory containing TensorBoard logs.
        port (int): Port to run TensorBoard on (default: 6006).
    """
    try:
        # Kill any existing tensorboard processes (cross-platform)
        if platform.system() == "Windows":
            subprocess.run(
                ["taskkill", "/F", "/IM", "tensorboard.exe"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
        else:  # Linux / macOS
            subprocess.run(["pkill", "-f", "tensorboard"], check=False)

        # Start TensorBoard
        tb_process = subprocess.Popen(
            ["tensorboard", f"--logdir={logdir}", f"--port={port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Give TensorBoard a moment to start
        time.sleep(3)

        # Open browser automatically
        url = f"http://localhost:{port}"
        webbrowser.open(url)

        print(f"TensorBoard launched at {url}")
        return tb_process

    except FileNotFoundError:
        print("TensorBoard not found. Please install it with `pip install tensorboard`.")
        return None


# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------
# Device config
# -----------------------------
# Use GPU if available (important for training speed!)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")

# -----------------------------
# Block 2: Walk-forward validation + Experiment Logging
# -----------------------------


def walk_forward_split(X, y, train_size=0.8):
    """
    Perform a walk-forward split for time series.

    Args:
        X (np.array): Feature sequences of shape (samples, window, features).
        y (np.array): Target array of shape (samples,).
        train_size (float): Fraction of data to use for training (0 < train_size < 1).

    Returns:
        (X_train, y_train, X_val, y_val)
    """
    split_idx = int(len(X) * train_size)

    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    return X_train, y_train, X_val, y_val


def get_tensorboard_writer(log_dir="runs"):
    """
    Create a TensorBoard writer for experiment logging.

    Args:
        log_dir (str): Base directory for logs.

    Returns:
        SummaryWriter object
    """
    # Add timestamped subfolder for each run
    timestamp = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_path = f"{log_dir}/run_{timestamp}"

    writer = SummaryWriter(log_dir=run_path)
    logging.info(f"TensorBoard logging started at {run_path}")

    return writer


# -----------------------------
# Block 3: Model Definition + Dataset Preparation
# -----------------------------


class StockPredictor(nn.Module):
    """
    Simple LSTM-based model for stock prediction using technical indicators.
    Future improvements:
        - Add GRU / Transformer layers.
        - Multi-ticker input.
        - Attention mechanism for feature importance.
    """

    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(StockPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # out: (batch, seq_len, hidden_size)
        out = out[:, -1, :]  # take last time step
        out = self.fc(out)   # map to output
        return out.squeeze(-1)


# -----------------------------
# Block 4: Training Loop
# -----------------------------

def train_model(X_train, y_train, X_val, y_val, input_size,
                epochs=20, batch_size=16, lr=1e-3, writer=None, scaler=None,
                early_stopping_patience=20, lr_scheduler_patience=5, lr_scheduler_factor=0.5):
    """
    Train the LSTM model with walk-forward validation, TensorBoard logging,
    early stopping, and learning rate scheduling.

    Args:
        X_train, y_train, X_val, y_val: numpy arrays
        input_size (int): number of features per timestep
        epochs (int): maximum number of training epochs
        batch_size (int): batch size
        lr (float): initial learning rate
        writer (SummaryWriter): optional TensorBoard writer
        scaler: MinMaxScaler for inverse transformation (optional)
        early_stopping_patience (int): stop if no improvement for N epochs (default: 10)
        lr_scheduler_patience (int): reduce LR if no improvement for N epochs (default: 5)
        lr_scheduler_factor (float): factor to reduce LR by (default: 0.5)

    Returns:
        model: trained LSTM model with best weights loaded
    """
    model = StockPredictor(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        output_size=1,
        dropout=0.2
    ).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Learning Rate Scheduler
    # Reduces LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # Minimize validation loss
        factor=lr_scheduler_factor,  # Multiply LR by this when reducing
        patience=lr_scheduler_patience,  # Wait N epochs before reducing
        min_lr=1e-6  # Don't go below this LR
    )

    # Convert data to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_val_loss = float("inf")
    checkpoint_path = "best_model.pth"

    # Early stopping variables
    epochs_without_improvement = 0
    early_stop = False

    for epoch in range(1, epochs + 1):
        # --- Training Phase ---
        model.train()
        train_losses = []

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # --- Validation Phase ---
        model.eval()
        val_losses = []
        y_pred_list, y_true_list = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_X)
                val_loss = criterion(outputs, batch_y)
                val_losses.append(val_loss.item())

                y_pred_list.append(outputs.detach().cpu().view(-1))
                y_true_list.append(batch_y.detach().cpu().view(-1))

        avg_val_loss = np.mean(val_losses)
        y_pred = torch.cat(y_pred_list)
        y_val_tensor = torch.cat(y_true_list)

        # Calculate percentage error (scale-independent)
        mape = torch.mean(torch.abs((y_pred - y_val_tensor) / (y_val_tensor + 1e-8))) * 100

        # Calculate prediction error (always, not just for tensorboard)
        returns = y_pred.numpy() - y_val_tensor.numpy()
        prediction_error = float(np.mean(returns) / (np.std(returns) + 1e-8))

        # --- Learning Rate Scheduling ---
        # Update scheduler based on validation loss
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        # Check if LR was reduced
        if new_lr < current_lr:
            logging.info(f"Learning rate reduced: {current_lr:.6f} → {new_lr:.6f}")

        # --- TensorBoard Logging ---
        if writer:
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("Loss/Val", avg_val_loss, epoch)
            writer.add_scalar("Metrics/MAPE_Percent", mape.item(), epoch)
            writer.add_scalar("Metrics/PredictionError", prediction_error, epoch)
            writer.add_scalar("Training/LearningRate", current_lr, epoch)
            writer.add_scalar("Training/EpochsWithoutImprovement", epochs_without_improvement, epoch)

        # --- Console Logging ---
        logging.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"MAPE: {mape:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # --- Save Best Model & Early Stopping Check ---
        if avg_val_loss < best_val_loss:
            # Improvement! Reset counter
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Best model saved (val_loss: {best_val_loss:.6f})")
        else:
            # No improvement
            epochs_without_improvement += 1
            logging.info(f"No improvement for {epochs_without_improvement} epoch(s)")

            # Check if we should stop
            if epochs_without_improvement >= early_stopping_patience:
                logging.info(f"\n{'=' * 60}")
                logging.info(f"Early stopping triggered after {epoch} epochs")
                logging.info(
                    f"Best validation loss: {best_val_loss:.6f} (at epoch {epoch - epochs_without_improvement})")
                logging.info(f"{'=' * 60}\n")
                early_stop = True
                break

    # --- Training Complete ---
    if not early_stop:
        logging.info("Training complete.")

    # Load the best model checkpoint before returning
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        logging.info(f"Loaded best model from {checkpoint_path} (val_loss: {best_val_loss:.6f})")
    else:
        logging.warning("Checkpoint not found. Returning final epoch model.")

    return model


# -----------------------------
# Block 5: Orchestration / Run training for a ticker
# -----------------------------

def run_training_for_ticker(
    ticker,
    data_dir=None,
    window_size=20,
    feature_columns=None,
    target_column="close",
    rsi_period=14,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    sma_period=20,
    train_size=0.8,
    epochs=50,
    batch_size=32,
    lr=1e-3,
    early_stopping_patience=10,
    lr_scheduler_patience=5,
    lr_scheduler_factor=0.5,
    model_dir="models",
    writer=None
    ):
    """
    High level helper to prepare data for a ticker, run train_model, and save artifacts.

    Args:
        ticker (str): ticker symbol.
        data_dir (str|None): path to raw CSVs (None -> auto-locate).
        window_size (int): sequence length for prepare_data_for_ai.
        feature_columns (list|None): which columns to use; None -> defaults.
        target_column (str)
        rsi_period, macd_*, sma_period: indicator parameters forwarded to prepare_data_for_ai.
        train_size (float): fraction used for training in walk_forward_split.
        epochs, batch_size, lr: training hyperparameters.
        model_dir (str): directory to save trained model + scaler.
        writer (SummaryWriter|None): TensorBoard writer (optional).

    Returns:
        dict with metadata: {'model_path': ..., 'scaler_path': ..., 'best_val_loss': ...}
    """
    logging.info(f"Preparing data for {ticker} (window={window_size})...")
    X, y, scaler = prepare_data_for_ai(
        ticker,
        data_dir=data_dir,
        feature_columns=feature_columns,
        target_column=target_column,
        window_size=window_size,
        rsi_period=rsi_period,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        sma_period=sma_period
    )

    if X is None or y is None or len(X) == 0:
        logging.error(f"Data preparation failed or empty for {ticker}. Aborting training.")
        return None

    if len(X) < window_size:
        logging.error(f"Not enough data: need >{window_size}, got {len(X)}")
        return None

    # Walk-forward split
    X_train, y_train, X_val, y_val = walk_forward_split(X, y, train_size=train_size)
    logging.info(f"Data split: train={len(X_train)} samples, val={len(X_val)} samples")

    # If caller didn't provide a writer, create one for this run
    own_writer = False
    if writer is None:
        writer = get_tensorboard_writer()
        own_writer = True

    # Train
    logging.info(f"Starting training for {ticker} for {epochs} epochs...")
    model = train_model(
        X_train, y_train, X_val, y_val,
        input_size=X_train.shape[2],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        writer=writer,
        scaler=scaler,
        early_stopping_patience = early_stopping_patience,
        lr_scheduler_patience = lr_scheduler_patience,
        lr_scheduler_factor = lr_scheduler_factor
    )

    # Close writer if we created it
    if own_writer:
        writer.close()

    # Model artifact handling
    # train_model saves 'best_model.pth' when it finds a better validation loss.
    default_checkpoint = Path("best_model.pth")
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    timestamp = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
    model_filename = f"{ticker}_model_{timestamp}.pth"
    model_path = Path(model_dir) / model_filename

    if default_checkpoint.exists():
        try:
            shutil.move(str(default_checkpoint), str(model_path))
            logging.info(f"Saved trained model to {model_path}")
        except Exception as e:
            logging.error(f"Failed to move checkpoint: {e}")
            model_path = None
    else:
        logging.warning("Expected checkpoint 'best_model.pth' not found. No model saved.")
        model_path = None

    # Save scaler (so inference will use same scaling)
    scaler_filename = f"{ticker}_scaler_{timestamp}.joblib"
    scaler_path = Path(model_dir) / scaler_filename
    try:
        joblib.dump(scaler, scaler_path)
        logging.info(f"Saved scaler to {scaler_path}")
    except Exception as e:
        logging.error(f"Failed to save scaler: {e}")
        scaler_path = None


    return {
        "model_path": str(model_path) if model_path else None,
        "scaler_path": str(scaler_path) if scaler_path else None,
        "timestamp": timestamp
    }


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # ---- Launch TensorBoard automatically ----
    tb_process = launch_tensorboard(logdir="runs", port=6006)

    # ---- Test walk_forward_split ----
    X_dummy = np.random.rand(100, 20, 5)  # 100 samples, window=20, 5 features
    y_dummy = np.random.rand(100)

    X_train, y_train, X_val, y_val = walk_forward_split(X_dummy, y_dummy, train_size=0.8)

    assert len(X_train) == 80, f"Expected 80 train samples, got {len(X_train)}"
    assert len(y_train) == 80, f"Expected 80 train targets, got {len(y_train)}"
    assert len(X_val) == 20, f"Expected 20 validation samples, got {len(X_val)}"
    assert len(y_val) == 20, f"Expected 20 validation targets, got {len(y_val)}"

    logging.info("walk_forward_split passed all tests.")

    # ---- Test TensorBoard writer ----
    writer = get_tensorboard_writer()
    writer.add_scalar("Test/Loss", 0.123, 1)  # log dummy value
    writer.close()

    logging.info("TensorBoard writer test completed. Check 'runs/' folder for logs.")

    # ---- Train the LSTM model with dummy data ----
    logging.info("Starting dummy training loop...")
    writer = get_tensorboard_writer()  # reopen writer for training logs
    model = train_model(
        X_train, y_train, X_val, y_val,
        input_size=X_train.shape[2],  # number of features
        epochs=5,  # keep short for testing
        batch_size=16,
        lr=1e-3,
        writer=writer,
        scaler=None
    )
    writer.close()
    logging.info("Dummy training completed.")

    # ---- Keep TensorBoard alive until user stops script ----
    try:
        logging.info("TensorBoard running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        if tb_process:
            tb_process.terminate()
            logging.info("TensorBoard stopped.")