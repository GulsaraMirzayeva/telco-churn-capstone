import time
import tomllib
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECRETS_PATH = PROJECT_ROOT / ".streamlit" / "secrets.toml"

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco.csv"
TABLE_NAME = "telco_customer_lookup"


def load_database_url() -> str:
    print("Reading secrets.toml...")

    if not SECRETS_PATH.exists():
        raise FileNotFoundError(
            "Missing .streamlit/secrets.toml. Please add DATABASE_URL there first."
        )

    with open(SECRETS_PATH, "rb") as file:
        secrets = tomllib.load(file)

    database_url = secrets.get("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL was not found in .streamlit/secrets.toml")

    print("DATABASE_URL found.")
    return database_url


def create_db_engine(database_url: str):
    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args={
            "connect_timeout": 20,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        },
    )


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    PostgreSQL can handle quoted column names with spaces,
    but clean snake_case names make SQL lookup safer and easier.
    """
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    return df


def upload_chunk_with_retry(
    chunk: pd.DataFrame,
    database_url: str,
    table_name: str,
    chunk_label: str,
    max_retries: int = 3,
):
    for attempt in range(1, max_retries + 1):
        engine = None

        try:
            engine = create_db_engine(database_url)

            chunk.to_sql(
                table_name,
                engine,
                if_exists="append",
                index=False,
                method="multi",
            )

            print(f"{chunk_label} uploaded successfully.")
            return

        except Exception as error:
            print(f"{chunk_label} failed on attempt {attempt}/{max_retries}.")
            print(f"Error: {error}")

            if engine is not None:
                engine.dispose()

            if attempt == max_retries:
                raise

            print("Retrying in 5 seconds...")
            time.sleep(5)


def main():
    print("Starting Neon upload script...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print(f"Reading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset loaded. Original shape: {df.shape}")
    print("Original columns:")
    print(df.columns.tolist())

    df = clean_column_names(df)

    print(f"Cleaned shape: {df.shape}")
    print("Cleaned columns:")
    print(df.columns.tolist())

    database_url = load_database_url()
    engine = create_db_engine(database_url)

    print("Testing database connection...")
    with engine.connect() as connection:
        version = connection.execute(text("SELECT version();")).scalar()
        print("Connected to PostgreSQL.")
        print(version)

    print(f"Dropping old table if exists: {TABLE_NAME}")
    with engine.begin() as connection:
        connection.execute(text(f'DROP TABLE IF EXISTS "{TABLE_NAME}"'))

    print(f"Creating empty table: {TABLE_NAME}")
    df.head(0).to_sql(
        TABLE_NAME,
        engine,
        if_exists="replace",
        index=False,
    )

    engine.dispose()

    print(f"Uploading data to table: {TABLE_NAME}")

    chunk_size = 200
    total_rows = len(df)

    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        chunk = df.iloc[start:end].copy()

        chunk_label = f"Rows {start + 1} to {end} of {total_rows}"

        upload_chunk_with_retry(
            chunk=chunk,
            database_url=database_url,
            table_name=TABLE_NAME,
            chunk_label=chunk_label,
            max_retries=3,
        )

    print("Upload finished. Creating index on customer_id...")

    engine = create_db_engine(database_url)

    with engine.begin() as connection:
        connection.execute(
            text(
                f'CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_customer_id '
                f'ON "{TABLE_NAME}" (customer_id)'
            )
        )

    print("Checking row count...")

    with engine.connect() as connection:
        row_count = connection.execute(
            text(f'SELECT COUNT(*) FROM "{TABLE_NAME}"')
        ).scalar()

    engine.dispose()

    print(f"Uploaded table: {TABLE_NAME}")
    print(f"Rows uploaded: {row_count}")
    print("Upload completed successfully.")


if __name__ == "__main__":
    main()