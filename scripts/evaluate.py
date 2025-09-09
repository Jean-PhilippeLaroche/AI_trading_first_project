import os
import logging
import torch
import numpy as np
import torch.nn as nn
import pandas as pd

logging.basicConfig(level=logging.INFO)


def evaluate_model(model, X_val, y_val, device="cpu"):
    """
    Evaluate a trained model on validation data.

    Args:
        model: Trained PyTorch model
        X_val: Validation input data (numpy or torch tensor)
        y_val: Validation targets
        device: "cpu" or "cuda"

    Returns:
        val_loss: Average validation loss
        predictions: Model predictions on X_val
    """
    model.to(device)
    model.eval()

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(X_val_tensor).squeeze()
        val_loss = torch.nn.functional.mse_loss(predictions, y_val_tensor)

    return val_loss.item(), predictions.cpu().numpy()

# ---- TODOs for future implementation ----
# 1. Add functionality to log evaluation metrics to TensorBoard
# 2. Compute additional metrics like Sharpe ratio, max drawdown
# 3. Generate plots for predictions vs actual using plot_utils



if __name__ == "__main__":
    logging.info("Starting dummy evaluation test...")

    # ---- Create dummy validation data ----
    X_val = np.random.rand(20, 10, 5)  # 20 samples, 10 timesteps, 5 features
    y_val = np.random.rand(20)

    # ---- Define a simple dummy model ----
    class DummyModel(nn.Module):
        def __init__(self, input_size=5, hidden_size=16):
            super(DummyModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            out = self.fc(h_n[-1])
            return out

    model = DummyModel()

    # ---- Evaluate the dummy model ----
    val_loss, predictions = evaluate_model(model, X_val, y_val, device="cpu")

    logging.info(f"Dummy evaluation completed. Validation loss: {val_loss:.4f}")
    logging.info(f"Predictions shape: {predictions.shape}")
    logging.info("Test passed if predictions shape matches y_val length.")