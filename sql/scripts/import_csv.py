import pandas as pd
from sqlalchemy import create_engine

# 1. Database Credentials
DB_USER = "joshuaxu"
DB_PASSWORD = ""
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "loans_db"

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

csv_file_path = "/Users/joshuaxu/Desktop/loan_outcome_predictor/data/lending_club_loans/LendingClub_2007_to_2018Q4.csv"

print("Reading CSV and auto-creating table structure...")

chunksize = 100000
table_name = "loans"

for i, chunk in enumerate(pd.read_csv(csv_file_path, chunksize=chunksize, low_memory=False, dtype=str)):
    # 1. Force the 'id' column to numeric; non-numeric text like footers become 'NaN'
    chunk['id'] = pd.to_numeric(chunk['id'], errors='coerce')
    
    # 2. Drop any summary/footer rows where 'id' is NaN
    chunk = chunk.dropna(subset=['id'])
    
    # 3. Determine if we are creating or appending to the table
    mode = "replace" if i == 0 else "append"
    
    # Write to PostgreSQL
    chunk.to_sql(table_name, engine, if_exists=mode, index=False)
    print(f"Processed chunk {i + 1}...")

print("Import complete! Your table 'loans' is ready in loans_db.")