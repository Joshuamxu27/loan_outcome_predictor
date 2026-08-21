from pathlib import Path
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('postgresql://joshuaxu@localhost:5432/loans_db')

sql_path = Path('sql/queries/training_data_query.sql')
sql_query = sql_path.read_text()

df = pd.read_sql(sql_query, engine)

df.to_parquet('data/ML_training_data/model_training_data_raw.parquet', index = False)