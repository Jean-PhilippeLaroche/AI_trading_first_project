# Path
No hardcoded absolute paths for exporting(like C:/Users/...).
os.path.join(BASE_DIR, ...) so project works on any machine
Example:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data/raw")

--- Train.py ---
if __name__ == "__main__" tests run on dummy data
Useful for visualisation of tensorboard metrics change or training changes


# Environment management
Export dependencies:
pip freeze > requirements.txt
On other machines, install:
pip install -r requirements.txt
PyTorch version works with cpu, not cuda so need to install it manually


# Logging & outputs
All logs, models, and outputs are inside the project folder (e.g., logs/ or models/).


# Version control
Using Git to keep the code synced.
Raw data tracked as well for now, small enough


# Command for multi ticker history price download for raw data:
python scripts/download_data_polygon.py --tickers AAPL MSFT AMZN JPM BAC XOM CAT WMT KO TSLA --start 2020-11-01 --end 2025-11-01 --interval 1m --output_dir data/raw


# How top open Tensorboard log files under runs folder:
1. tensorboard --logdir=runs indide project .venv terminal
2. go to browser and type http://localhost:6006
Note: added automatic opening, use manual way if needed shouldn't be the case

# Run Commands
 - Larger model for better performance
python main.py --ticker AAPL --d_model 256 --nhead 8 --num_layers 4 --dim_feedforward 1024

 - Smaller model for faster training
python main.py --ticker AAPL --d_model 64 --nhead 4 --num_layers 2 --dim_feedforward 256

 - Adjust training parameters
python main.py --ticker AAPL --epochs 30 --batch 128 --lr 5e-5

# Hypertuning parameters

# TO DO:
1. Add forward testing with the best_model.pth
2. Keep on working on the project, add more advanced functionnalities