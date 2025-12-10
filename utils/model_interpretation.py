import torch
from scripts.train import TimeSeriesTransformerPooled
from utils.data_utils import prepare_data_for_ai, add_indicators, clean_data
import logging
from utils.data_utils import load_stock_csv
import os
import numpy as np
import json


def find_file(filename, start_dir=None, max_levels=5):
    if start_dir is None:
        start_dir = os.path.dirname(os.path.abspath(__file__))

    current = start_dir

    for _ in range(max_levels):
        candidate = os.path.join(current, filename)
        if os.path.exists(candidate):
            return candidate
        current = os.path.dirname(current)

    raise FileNotFoundError(f"{filename} not found up to {max_levels} levels above {start_dir}")


def model_interpretation(
        ticker="AAPL", train_size=0.8, window_size=20, d_model=64, nhead=4, num_layers=2
    ):

    df_raw = load_stock_csv(ticker)
    if df_raw is None:
        logging.error("Could not load raw CSV for ticker; exiting.")

    df_tmp = add_indicators(df_raw)
    df_tmp = clean_data(df_tmp)

    n_total = len(df_tmp)
    split_idx = int(n_total * train_size)

    X_train, y_train, scaler = prepare_data_for_ai(
            ticker=ticker,
            data_dir=None,
            feature_columns=None,
            target_column="close",
            window_size=window_size,
            start_idx=0,
            end_idx=split_idx
        )
    input_size = X_train.shape[2]
    d_model = 64
    nhead = 4
    num_layers = 2
    dim_feedforward = 256

    file_path = find_file("best_model.pth")

    model = TimeSeriesTransformerPooled(input_size=input_size, d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward)
    model.load_state_dict(torch.load(file_path, map_location="cpu"))
    model.eval()

    results_file = f'model_weights.json'
    params_json = {name: p.cpu().numpy().tolist() for name, p in model.state_dict().items()}
    with open(results_file, "w") as f:
        json.dump(params_json, f, indent=2)

if __name__ == "__main__":
    model_interpretation()
