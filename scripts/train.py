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
- Support multiple walk-forward folds (not just one train/val split).
- Integrate wandb for richer experiment tracking if needed.
- Add multi-horizon forecasting (predict multiple days ahead).
- Add feature-wise attention / embedding layers for indicators.
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
import math
from utils.transformer_visuals import update_attention_window, init_attention_window


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
# Block 3: Transformer Model Definition
# -----------------------------

class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings.
    Since Transformers have no inherent sense of order (unlike RNNs),
    we need to inject position information.
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Use sine and cosine functions of different frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeSeriesTransformerPooled(nn.Module):
    """
    Variant that uses both mean and max pooling over the sequence
    instead of just taking the last timestep.
    Often more robust for capturing overall trends.
    """

    def __init__(
            self,
            input_size,
            d_model=128,
            nhead=8,
            num_layers=3,
            dim_feedforward=512,
            dropout=0.1,
            max_len=5000

    ):
        super().__init__()

        self.return_attn = True  # allow extraction for transformer_visuals

        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # Output projection takes 2*d_model (mean + max pooling concatenated)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # Project and add positional encoding
        x = self.input_projection(x)
        x = self.pos_encoder(x)

        # to store attention values for viz
        all_attn = []

        # Manually forward through layers to access attention weights
        for layer in self.transformer_encoder.layers:
            x_before = x
            x2, attn = layer.self_attn(
                x_before,
                x_before,
                x_before,
                need_weights=self.return_attn,
                average_attn_weights=False
            )
            all_attn.append(attn)  # shape: (batch, heads, seq, seq)

            # Continue through feedforward part
            x = layer.norm1(x_before + x2)
            x = layer.norm2(x + layer.linear2(layer.dropout(layer.activation(layer.linear1(x)))))

        # Pool over sequence dimension
        # Mean pooling captures average pattern
        mean_pool = torch.mean(x, dim=1)  # (batch, d_model)
        # Max pooling captures strongest signals
        max_pool, _ = torch.max(x, dim=1)  # (batch, d_model)

        # Concatenate both pooling strategies
        x = torch.cat([mean_pool, max_pool], dim=1)  # (batch, 2*d_model)

        # Output projection
        output = self.output_projection(x)

        if self.return_attn:
            return output, all_attn
        else:
            return output


# -----------------------------
# Block 4: Training Loop
# -----------------------------

def train_model(X_train, y_train, X_val, y_val, input_size,
                epochs=20, batch_size=64, lr=1e-4, writer=None, scaler=None,
                early_stopping_patience=20, lr_scheduler_patience=5, lr_scheduler_factor=0.5,
                d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1):
    """
    Train the Transformer model.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        input_size: Number of input features
        epochs: Number of training epochs
        batch_size: Batch size (can be larger for Transformers)
        lr: Learning rate (lower for Transformers, typically 1e-4)
        writer: TensorBoard writer
        scaler: Data scaler (for inverse transform)
        early_stopping_patience: Early stopping patience
        lr_scheduler_patience: LR scheduler patience
        lr_scheduler_factor: LR scheduler reduction factor
        d_model: Transformer model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Feedforward dimension
        dropout: Dropout rate
    """

    # Initialize Transformer model
    model = TimeSeriesTransformerPooled(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(DEVICE)

    init_attention_window(num_layers, nhead, X_train.shape[1])

    logging.info(f"Initialized Transformer with {sum(p.numel() for p in model.parameters()):,} parameters")

    criterion = nn.MSELoss()

    # AdamW optimizer (better for Transformers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Learning rate scheduler
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

            outputs, attn = model(batch_X)
            outputs = outputs.squeeze(-1)

            # Compute mean attention map across batch
            mean_attn = [a.mean(dim=0).detach().cpu().numpy() for a in attn]
            # mean_attn becomes a list: [ (heads, seq, seq), ... per layer ]

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

                outputs, _ = model(batch_X)
                outputs = outputs.squeeze(-1)

                val_loss = criterion(outputs, batch_y)
                val_losses.append(val_loss.item())

        val_loop_times.append(time.time() - val_loop_start)

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        scheduler.step(avg_val_loss)

        epoch_times.append(time.time() - epoch_start)

        update_attention_window(mean_attn, epoch)

        # --------------------------
        # TENSORBOARD LOGGING
        # --------------------------
        if writer is not None:
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Loss/validation', avg_val_loss, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # --------------------------
        # CONSOLE LOGGING
        # --------------------------
        logging.info(f"\nEpoch {epoch}/{epochs}")
        print(f"  Train Loop: {train_loop_times[-1]:.3f}s")
        print(f"    Batch loading: {ep_batch_load:.2f}s")
        print(f"    Forward pass : {ep_forward:.2f}s")
        print(f"    Backward pass: {ep_backward:.2f}s")
        print(f"    Optimizer    : {ep_optimizer:.2f}s")
        print(f"  Validation Loop:   {val_loop_times[-1]:.3f}s")
        print(f"  Total:      {epoch_times[-1]:.3f}s")
        print(f"  Loss:       {avg_train_loss:.6f} (train), {avg_val_loss:.6f} (val)")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.2e}")

        # --------------------------
        # Early Stopping
        # --------------------------
        if avg_val_loss < best_val_loss:
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"New best model saved (val_loss: {avg_val_loss:.6f})")
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"Epochs without improvement: {epochs_without_improvement}")
            if epochs_without_improvement >= early_stopping_patience:
                logging.info(
                    f"\nEarly stopping triggered after {epochs_without_improvement} epochs without improvement")
                break

    # --------------------------
    # LOAD BEST MODEL
    # --------------------------
    model.load_state_dict(torch.load(checkpoint_path))
    logging.info(f"Loaded best model from {checkpoint_path}")

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

    # ---- Train the Transformer model with dummy data ----
    logging.info("Starting dummy training loop...")
    writer = get_tensorboard_writer()  # reopen writer for training logs
    model = train_model(
        X_train, y_train, X_val, y_val,
        input_size=X_train.shape[2],  # number of features
        epochs=5,  # keep short for testing
        batch_size=32,  # Larger batch for Transformer
        lr=1e-4,  # Lower LR for Transformer
        writer=writer,
        scaler=None,
        d_model=64,  # Smaller for testing
        nhead=4,
        num_layers=2
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