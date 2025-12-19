import pandas as pd
import sqlite3
from model_interpretation import find_file

# Connect to SQLite
database_path = find_file("data.db")
conn = sqlite3.connect(database_path)


def migrate_to_SQLite(csv_files: list):
    """
    Migrates a list of CSV files to an SQLite database.
    Each CSV file becomes a table with the same name (without .csv).
    """
    for file_name in csv_files:
        # Find the file path
        file_path = find_file(file_name)
        if file_path is None:
            print(f"File {file_name} not found, skipping.")
            continue

        # Read CSV into DataFrame
        df = pd.read_csv(file_path)

        # Use the filename (without extension) as table name
        table_name = file_name.replace(".csv", "")

        # Write DataFrame to SQLite
        df.to_sql(name=table_name, con=conn, if_exists="replace", index=False)
        print(f"{file_name} -> table '{table_name}' migrated successfully.")


if __name__ == "__main__":
    # List of CSV files to migrate
    raw_data = ["AAPL.csv", "MSFT.csv"]

    migrate_to_SQLite(raw_data)
    conn.close()
    print("All CSV files migrated to SQLite.")