# Loan Outcome Predictor — Database & SQL Analysis

A PostgreSQL-based data project analyzing 2.6M+ loan records (40 features), focused on database design, SQL querying, and data cleaning at scale.

## Skills Demonstrated
- Database creation & schema design (PostgreSQL)
- Writing and optimizing SQL queries on a 2.6M-row table
- Data exploration using SQL (COUNT, GROUP BY, aggregate functions, information_schema)
- Connecting SQL databases to Python (SQLAlchemy/psycopg2) for analysis with Pandas

## Tech Stack
- PostgreSQL
- Python (pandas, SQLAlchemy, psycopg2) — used to connect to the database for analysis
- Git, Github

## Database Setup
- Created a PostgreSQL database (`loans_db`) with a `loans` table containing 40 features
- Loaded 2.6 million rows from [source] into Postgres
- Table schema defined in `schema.sql`

## Data Import

The raw dataset (2.6M+ rows) was imported from CSV into PostgreSQL using a Python script (`import_csv_to_db.py`) rather than a direct SQL `COPY`, in order to handle chunked loading and clean up malformed rows during import.

Key steps handled during import:
- Read the CSV in chunks of 100,000 rows to manage memory usage
- Coerced the `id` column to numeric, converting invalid/non-numeric entries (e.g. footer/summary rows) to null
- Dropped rows where `id` was null, removing footer/junk rows from the raw CSV
- Wrote the first chunk with `if_exists="replace"` to create the table, then appended subsequent chunks

\`\`\`python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

chunksize = 100000
table_name = "loans"

for i, chunk in enumerate(pd.read_csv(csv_file_path, chunksize=chunksize, low_memory=False, dtype=str)):
    chunk['id'] = pd.to_numeric(chunk['id'], errors='coerce')
    chunk = chunk.dropna(subset=['id'])
    
    mode = "replace" if i == 0 else "append"
    chunk.to_sql(table_name, engine, if_exists=mode, index=False)
\`\`\`

Full script available in [`import_csv_to_db.py`](./import_csv_to_db.py).

## Future Work
- Expand into feature engineering and predictive modeling (e.g., logistic regression, XGBoost) to predict loan default risk
- Explore more advanced SQL techniques (window functions, CTEs, indexing for performance)
- Potentially move data cleaning steps into stored procedures or a more automated pipeline