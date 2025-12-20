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


if __name__ == "__main__":
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