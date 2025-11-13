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
from torch.utils.tensorboard import SummaryWriter
import datetime
import subprocess
import webbrowser
import time
import platform
import uuid

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

    model = StockPredictor(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        output_size=1,
        dropout=0.2
    ).to(DEVICE)


    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_scheduler_factor,
        patience=lr_scheduler_patience, min_lr=1e-6
    )

    # Dataset & DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_val_loss = float("inf")
    checkpoint_path = "best_model.pth"
    epochs_without_improvement = 0

    # -----------------------
    # PROFILING DATA STORAGE
    # -----------------------
    epoch_times = []
    train_loop_times = []
    val_loop_times = []
    batch_load_times = []
    forward_times = []
    backward_times = []
    optimizer_times = []

    print("\n=== TRAINING PROFILER ===")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # --------------------------
        # TRAINING LOOP
        # --------------------------
        model.train()
        train_losses = []

        train_loop_start = time.time()

        # Per-epoch temporary accumulators
        ep_batch_load = 0.0
        ep_forward = 0.0
        ep_backward = 0.0
        ep_optimizer = 0.0

        for batch_X, batch_y in train_loader:

            t0 = time.time()
            batch_load_start = time.time()
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            batch_load_times.append(time.time() - batch_load_start)

            load_time = time.time() - t0
            ep_batch_load += load_time

            # Forward pass
            t0 = time.time()
            fwd_start = time.time()
            outputs = model(batch_X)
            forward_times.append(time.time() - fwd_start)

            fwd_time = time.time() - t0
            ep_forward += fwd_time

            loss = criterion(outputs, batch_y)

            # Backward pass
            t0 = time.time()
            bwd_start = time.time()
            loss.backward()
            backward_times.append(time.time() - bwd_start)

            bwd_time = time.time() - t0
            ep_backward += bwd_time

            # Optimizer
            t0 = time.time()
            opt_start = time.time()
            optimizer.step()
            optimizer.zero_grad()
            optimizer_times.append(time.time() - opt_start)

            opt_time = time.time() - t0
            ep_optimizer += opt_time

            train_losses.append(loss.item())


        train_loop_times.append(time.time() - train_loop_start)

        batch_load_times.append(ep_batch_load)
        forward_times.append(ep_forward)
        backward_times.append(ep_backward)
        optimizer_times.append(ep_optimizer)

        # --------------------------
        # VALIDATION LOOP
        # --------------------------
        model.eval()
        val_loop_start = time.time()
        val_losses = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_X)
                val_loss = criterion(outputs, batch_y)
                val_losses.append(val_loss.item())

        val_loop_times.append(time.time() - val_loop_start)

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        scheduler.step(avg_val_loss)

        epoch_times.append(time.time() - epoch_start)

        # --------------------------
        # LOGGING
        # --------------------------
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  Train Loop: {train_loop_times[-1]:.3f}s")
        print(f"    Batch loading: {ep_batch_load:.2f}s")
        print(f"    Forward pass : {ep_forward:.2f}s")
        print(f"    Backward pass: {ep_backward:.2f}s")
        print(f"    Optimizer    : {ep_optimizer:.2f}s")
        print(f"  Validation Loop:   {val_loop_times[-1]:.3f}s")
        print(f"  Total:      {epoch_times[-1]:.3f}s")
        print(f"  Loss:       {avg_train_loss:.6f} (train), {avg_val_loss:.6f} (val)")

        # --------------------------
        # Early Stopping
        # --------------------------
        if avg_val_loss < best_val_loss:
            torch.save(model.state_dict(), checkpoint_path)
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epochs_without_improvement} epochs without improvements")
                break

    # --------------------------
    # FINAL PROFILING SUMMARY
    # --------------------------
    print("\n=== TRAINING PROFILING SUMMARY ===")
    print(f"Epoch avg time:     {np.mean(epoch_times):.3f}s")
    print(f"Train loop avg:     {np.mean(train_loop_times):.3f}s")
    print(f"Val loop avg:       {np.mean(val_loop_times):.3f}s")

    print("\n--- Batch timings ---")
    print(f"Avg batch load:     {np.mean(batch_load_times):.6f}s")
    print(f"Avg forward pass:   {np.mean(forward_times):.6f}s")
    print(f"Avg backward pass:  {np.mean(backward_times):.6f}s")
    print(f"Avg optimizer step: {np.mean(optimizer_times):.6f}s")

    print("\n--- Bottleneck Hint ---")
    if np.mean(forward_times) > np.mean(backward_times) * 1.5:
        print("Forward pass is slow → model too big, window too large, or GPU underutilized.")
    if np.mean(batch_load_times) > np.mean(forward_times):
        print("DataLoader is the bottleneck → num_workers, pin_memory.")
    if np.mean(backward_times) > np.mean(forward_times) * 1.5:
        print("Backward pass dominates → reduce hidden size or num_layers.")
    if np.mean(optimizer_times) > 0.5 * np.mean(backward_times):
        print("Optimizer overhead → switch to fused/AdamW, or larger batch size.")

    return model



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