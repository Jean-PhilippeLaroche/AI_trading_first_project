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
python scripts/download_data.py --tickers AAPL MSFT AMZN JPM BAC XOM CAT WMT KO TSLA --start 2020-01-01 --end 2025-01-01 --interval 1d --output_dir data/raw


--- How top open Tensorboard log files under runs folder: ---
1. tensorboard --logdir=runs indide project .venv terminal
2. go to browser and type http://localhost:6006
Note: added automatic opening, use manual way if needed shouldn't be the case


--- Smoke run for test:
python main.py --ticker AAPL --epochs 10 --batch 8 --window 10 --train_size 0.7 --threshold 0.05 --balance 10000 --no_viz
Remove --no_viz for tensorboard and matplotlib graphs

-- Real run example:
python main.py --ticker AAPL --epochs 20 --batch 16 --window 20 --train_size 0.8 --threshold 0.05 --balance 10000

-- Big run example:
python main.py --ticker AAPL --epochs 200 --batch 64 --window 60 --train_size 0.9 --threshold 0.02 --balance 100000

---- TO DO: ----
1. Complete evaluate.py using other metrics
2. Complete plot_utils with everything I want to include as a tracker, matplotlib lab and etc.
3. Complete main.py
4. Keep on working on the project, add more advanced functionnalities