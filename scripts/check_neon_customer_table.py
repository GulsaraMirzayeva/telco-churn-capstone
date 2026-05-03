import tomllib
from pathlib import Path

from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECRETS_PATH = PROJECT_ROOT / ".streamlit" / "secrets.toml"
TABLE_NAME = "telco_customer_lookup"


def load_database_url() -> str:
    with open(SECRETS_PATH, "rb") as file:
        secrets = tomllib.load(file)

    return secrets["DATABASE_URL"]


def main():
    database_url = load_database_url()
    engine = create_engine(database_url, pool_pre_ping=True)

    with engine.connect() as connection:
        row_count = connection.execute(
            text(f'SELECT COUNT(*) FROM "{TABLE_NAME}"')
        ).scalar()

        sample_rows = connection.execute(
            text(
                f"""
                SELECT customer_id, contract, internet_type, monthly_charge, churn_label
                FROM "{TABLE_NAME}"
                LIMIT 5
                """
            )
        ).fetchall()

    print(f"Table: {TABLE_NAME}")
    print(f"Rows: {row_count}")
    print("Sample rows:")

    for row in sample_rows:
        print(row)


if __name__ == "__main__":
    main()