import os
import logging
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from utils.plot_utils import (
    plot_signals,
    plot_price_vs_prediction,
    plot_portfolio_equity,
    plot_indicator_overlay,
    plot_return_distribution
)

def visualize_model_performance(dates, actual_prices, predicted_prices, signals, portfolio_values, indicators=None):
    """
    Calls all plot_utils functions to visualize the model's performance.

    Args:
        dates (pd.DatetimeIndex or list): timestamps of the data
        actual_prices (np.ndarray): actual stock prices
        predicted_prices (np.ndarray): model predicted prices
        signals (np.ndarray): trading signals (+1 buy, -1 sell, 0 hold)
        portfolio_values (np.ndarray): equity value of portfolio over time
        indicators (dict of np.ndarray, optional): additional indicators to overlay
            e.g., {"SMA": sma_array, "RSI": rsi_array}
    """
    plot_signals(dates, actual_prices, signals)
    plot_price_vs_prediction(dates, actual_prices, predicted_prices)
    plot_portfolio_equity(portfolio_values, dates)
    if indicators:
        plot_indicator_overlay(actual_prices, indicators, dates)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    plot_return_distribution(returns)
    plt.show()



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

# validation tests for plot_utils being loaded in evaluate.py
    rng = pd.date_range("2023-01-01", periods=50, freq="D")
    actual_prices = np.linspace(100, 120, 50) + np.random.normal(0, 1, 50)
    predicted_prices = actual_prices + np.random.normal(0, 2, 50)
    signals = np.zeros(50)
    signals[10] = 1
    signals[25] = -1
    signals[40] = 1
    portfolio_values = np.cumsum(np.random.randn(50)) + 100  # fake equity curve
    indicators = {"SMA": np.convolve(actual_prices, np.ones(5) / 5, mode="same")}

    visualize_model_performance(rng, actual_prices, predicted_prices, signals, portfolio_values, indicators)