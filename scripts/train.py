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
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_path = f"{log_dir}/run_{timestamp}"

    writer = SummaryWriter(log_dir=run_path)
    logging.info(f"TensorBoard logging started at {run_path}")

    return writer


# -----------------------------
# TODOs for this block:
# - Support multiple walk-forward folds (not just one train/val split).
# - Log more than just loss: e.g., metrics like MAE, RMSE, R².
# - Save best model checkpoint to disk when val_loss improves.
# - Integrate wandb for richer experiment tracking if needed.
# -----------------------------

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
        return out


def prepare_dataloader(X, y, batch_size=32, shuffle=True):
    """
    Converts numpy arrays to PyTorch DataLoader for training/validation.

    Args:
        X (np.array): Feature sequences (samples, window, features)
        y (np.array): Targets (samples,)
        batch_size (int): Batch size for DataLoader
        shuffle (bool): Whether to shuffle dataset

    Returns:
        DataLoader object
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # ensure shape (samples, 1)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# -----------------------------
# TODOs for this block:
# - Add more complex architectures: GRU, Transformers.
# - Add multi-horizon forecasting (predict multiple days ahead).
# - Add feature-wise attention / embedding layers for indicators.
# - Support multi-ticker inputs as batch sequences.
# -----------------------------


# -----------------------------
# Block 4: Training Loop
# -----------------------------
# TODO:
# - Support multiple tickers by iterating over a list of tickers
# - Add early stopping
# - Add learning rate scheduler
# - Experiment with more advanced architectures (GRU, Transformer)
# - Integrate evaluation metrics and plotting once evaluate.py / plot_utils are implemented

class LSTMStockPredictor(nn.Module):
    """
    Simple LSTM model for stock prediction.
    Input: sequences of features (RSI, MACD, SMA, etc.)
    Output: single predicted price value
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMStockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last timestep
        out = self.fc(out)
        return out.squeeze()


def train_model(X_train, y_train, X_val, y_val, input_size,
                epochs=20, batch_size=16, lr=1e-3, writer=None):
    """
    Train the LSTM model with walk-forward validation and TensorBoard logging.

    Args:
        X_train, y_train, X_val, y_val: numpy arrays
        input_size (int): number of features per timestep
        epochs (int): number of training epochs
        batch_size (int)
        lr (float): learning rate
        writer (SummaryWriter): optional TensorBoard writer

    Returns:
        model: trained LSTM model
    """
    model = LSTMStockPredictor(input_size=input_size).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Convert data to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_val_loss = float("inf")
    checkpoint_path = "best_model.pth"

    for epoch in range(1, epochs + 1):
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

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_X)
                val_loss = criterion(outputs, batch_y)
                val_losses.append(val_loss.item())
        avg_val_loss = np.mean(val_losses)

        # TensorBoard logging
        if writer:
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("Loss/Val", avg_val_loss, epoch)

        logging.info(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Best model saved at epoch {epoch} with val_loss {best_val_loss:.6f}")

        # Placeholder: evaluate metrics once evaluate.py is implemented
        # TODO: metrics = evaluate_model(model, val_loader)
        # logging.info(f"Validation metrics: {metrics}")

        # Placeholder: plot predictions once plot_utils is implemented
        # TODO: plot_predictions(model, X_val, y_val)

    logging.info("Training complete.")
    return model


# -----------------------------
# Block 5: Orchestration / Run training for a ticker
# -----------------------------
import shutil
import joblib  # pip install joblib
from pathlib import Path

def run_training_for_ticker(
    ticker,
    data_dir=None,
    window_size=20,
    feature_columns=None,
    target_column="Close",
    rsi_period=14,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    sma_period=20,
    train_size=0.8,
    epochs=50,
    batch_size=32,
    lr=1e-3,
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

    if X is None or y is None:
        logging.error(f"Data preparation failed for {ticker}. Aborting training.")
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
        writer=writer
    )

    # Close writer if we created it
    if own_writer:
        writer.close()

    # Model artifact handling
    # train_model saves 'best_model.pth' when it finds a better validation loss.
    default_checkpoint = Path("best_model.pth")
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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

    # Placeholder: call evaluate.py once implemented
    # TODO: from evaluate import evaluate_saved_model
    # TODO: if model_path: metrics = evaluate_saved_model(model_path, scaler_path, ...)

    # Placeholder: plotting
    # TODO: from utils.plot_utils import plot_training_results
    # TODO: plot_training_results(...)

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
        writer=writer
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