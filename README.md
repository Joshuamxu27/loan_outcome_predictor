# Loan Outcome Predictor — Database & SQL Analysis

A PostgreSQL-based data project analyzing 2.6M+ loan records (40 features), focused on database design, SQL querying, and data cleaning at scale.

## Skills Demonstrated
- Database creation & schema design (PostgreSQL)
- Writing and optimizing SQL queries on a 2.6M-row table
- Data exploration using SQL (COUNT, GROUP BY, aggregate functions, information_schema)
- Data cleaning directly in SQL (filtering nulls, deduplication, type handling)
- Sampling large datasets efficiently (TABLESAMPLE)
- Connecting SQL databases to Python (SQLAlchemy/psycopg2) for downstream analysis

## Database Setup
- Created a PostgreSQL database (`loans_db`) with a `loans` table containing 40 features
- Loaded 2.6 million rows from [source] into Postgres
- Table schema defined in `schema.sql`

## Tech Stack
- PostgreSQL
- Python (pandas, SQLAlchemy, psycopg2) — used to connect to the database for analysis
- Git, Github

## Future Work
- Expand into feature engineering and predictive modeling (e.g., logistic regression, XGBoost) to predict loan default risk
- Explore more advanced SQL techniques (window functions, CTEs, indexing for performance)
- Potentially move data cleaning steps into stored procedures or a more automated pipeline