--- Path ---
No hardcoded absolute paths for exporting(like C:/Users/...).
os.path.join(BASE_DIR, ...) so project works on any machine
Example:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data/raw")


--- Bugs ---
In data_utils:
In create_sequences(), you scale the features but then try to predict the scaled target:
pythony.append(target[i + window_size]) 
Problem: Your y values are scaled (0-1), but in real-world usage, you'd want to predict actual prices. 
You'll need to inverse-transform predictions. 
Not necessarily a bug, but something to be aware of for model evaluation.

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
python download_data_polygon.py --tickers AAPL MSFT AMZN JPM BAC XOM CAT WMT KO TSLA --start 2020-11-01 --end 2025-11-01 --interval 1m --output_dir data/raw


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
python main.py --ticker AAPL --epochs 200 --batch 64 --window 60 --train_size 0.9 --threshold 0.02 --balance 10000
python main.py --ticker AAPL --epochs 200 --batch 16 --window 60 --train_size 0.9 --threshold 0.02 --balance 10000
---- TO DO: ----
1. Complete evaluate.py using other metrics
2. Complete plot_utils with everything I want to include as a tracker, matplotlib lab and etc.
3. Complete main.py
4. Keep on working on the project, add more advanced functionnalities