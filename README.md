void hardcoding absolute paths (like C:/Users/...).

Use os.path.join(BASE_DIR, ...) so your project works on any machine.

Example:

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data/raw")

3️⃣ Environment management

Keep a single Python virtual environment per machine or use conda environments.

Export dependencies:

pip freeze > requirements.txt


On the other machine, install:

pip install -r requirements.txt

4️⃣ Logging & outputs

Keep all logs, models, and outputs inside the project folder (e.g., logs/ or models/).

This avoids changing code when moving between machines.

5️⃣ Version control

Use Git to keep your code synced.

Data is usually not tracked by Git (use .gitignore for data/raw), but scripts, configs, and notebooks are synced automatically.

6️⃣ Optional: Network drive / cloud sync

If you want to run on either machine without moving files manually:

Sync data/ and models/ via OneDrive, Google Drive, or NAS.

Or use a small S3 bucket or local NAS share.


Command for multi ticker history price download:
python scripts/download_data.py --tickers AAPL MSFT AMZN JPM BAC XOM CAT WMT KO TSLA --start 2020-01-01 --end 2025-01-01 --interval 30m --output_dir data/raw

--- How top open Tensorboard log files under runs folder: ---
1. tensorboard --logdir=runs indide project .venv terminal
2. go to browser and type http://localhost:6006

---- TO DO: ----
1. Add more functionnalites to tensorboard in train.py
2. Complete evaluate.py using other metrics
3. Complete plot_utils with everything I want to include as a tracker, matplotlib lab and etc.
4. Complete main.py
5. Understand the output logic and how to use the model for future predictions