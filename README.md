--- Path ---
No hardcoded absolute paths for exporting(like C:/Users/...).
os.path.join(BASE_DIR, ...) so project works on any machine
Example:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data/raw")



--- Train.py ---
if __name__ == "__main__" tests run on dummy data
Useful for visualisation of tensorboard metrics change or training changes


--- Environment management ---
Export dependencies:
pip freeze > requirements.txt
On other machines, install:
pip install -r requirements.txt


--- Logging & outputs ---
All logs, models, and outputs are inside the project folder (e.g., logs/ or models/).


--- Version control ---
Using Git to keep the code synced.
Raw data tracked as well for now, small enough


--- Command for multi ticker history price download for raw data: ---
python scripts/download_data_polygon.py --tickers AAPL MSFT AMZN JPM BAC XOM CAT WMT KO TSLA --start 2020-11-01 --end 2025-11-01 --interval 1m --output_dir data/raw


--- How top open Tensorboard log files under runs folder: ---
1. tensorboard --logdir=runs indide project .venv terminal
2. go to browser and type http://localhost:6006
Note: added automatic opening, use manual way if needed shouldn't be the case

   
-- Run commands: ---
Basic: python main.py --ticker AAPL --epochs 50 --window 60
Custom: python main.py --ticker AAPL --epochs 200 --batch 64 --window 60 --train_size 0.8 --threshold 0.02 --balance 10000 --transaction_cost 0.02
No vizualisation: python main.py --ticker AAPL --epochs 50 --no_viz

--- hypertuning parameters, manual method
Custom: python main.py --ticker AAPL --epochs 200 --batch 64 --window 60 --train_size 0.8 --threshold 0.02 --balance 10000 --transaction_cost 0.02
First try: python main.py --ticker AAPL --epochs 50 --batch 128 --window 60 --train_size 0.8 --threshold 0.02 --balance 10000 --transaction_cost 0.02 -> trying to increase batch size first, sucks ass
Second try: python main.py --ticker AAPL --epochs 50 --batch 128 --window 60 --train_size 0.8 --threshold 0.02 --balance 10000 --transaction_cost 0.02
Third try: python main.py --ticker AAPL --epochs 50 --batch 32 --window 60 --train_size 0.8 --threshold 0.02 --balance 10000 --transaction_cost 0.02

---- TO DO: ----
1. Add forward testing with the best_model.pth
2. Keep on working on the project, add more advanced functionnalities