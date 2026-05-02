import sqlite3
from pathlib import Path

import pandas as pd


# -----------------------------
# 1. File paths
# -----------------------------
clean_data_path = Path("data/processed/bike_hour_clean.csv")
database_path = Path("sql/bike_sharing.db")
output_tables_path = Path("outputs/tables")

output_tables_path.mkdir(parents=True, exist_ok=True)


# -----------------------------
# 2. Load cleaned data
# -----------------------------
df = pd.read_csv(clean_data_path)

print("\nCleaned data loaded:")
print(df.head())
print("\nShape:")
print(df.shape)


# -----------------------------
# 3. Create SQLite database
# -----------------------------
conn = sqlite3.connect(database_path)

df.to_sql(
    "hourly_rentals",
    conn,
    if_exists="replace",
    index=False
)

print("\nData saved to SQLite table: hourly_rentals")


# -----------------------------
# 4. SQL queries
# -----------------------------

queries = {
    "rentals_by_hour": """
        SELECT
            hour,
            ROUND(AVG(total_rentals), 2) AS avg_total_rentals,
            ROUND(AVG(casual_rentals), 2) AS avg_casual_rentals,
            ROUND(AVG(registered_rentals), 2) AS avg_registered_rentals
        FROM hourly_rentals
        GROUP BY hour
        ORDER BY hour;
    """,

    "rentals_by_season": """
        SELECT
            season_name,
            ROUND(AVG(total_rentals), 2) AS avg_total_rentals,
            SUM(total_rentals) AS total_rentals
        FROM hourly_rentals
        GROUP BY season_name
        ORDER BY avg_total_rentals DESC;
    """,

    "rentals_by_weather": """
        SELECT
            weather_name,
            ROUND(AVG(total_rentals), 2) AS avg_total_rentals,
            SUM(total_rentals) AS total_rentals
        FROM hourly_rentals
        GROUP BY weather_name
        ORDER BY avg_total_rentals DESC;
    """,

    "workingday_vs_nonworkingday": """
        SELECT
            workingday,
            CASE
                WHEN workingday = 1 THEN 'working_day'
                ELSE 'non_working_day'
            END AS day_type,
            ROUND(AVG(total_rentals), 2) AS avg_total_rentals,
            SUM(total_rentals) AS total_rentals
        FROM hourly_rentals
        GROUP BY workingday
        ORDER BY workingday;
    """,

    "monthly_rental_trends": """
        SELECT
            year,
            month,
            month_name,
            SUM(total_rentals) AS total_rentals,
            ROUND(AVG(total_rentals), 2) AS avg_hourly_rentals
        FROM hourly_rentals
        GROUP BY year, month, month_name
        ORDER BY year, month;
    """,

    "top_10_highest_demand_hours": """
        SELECT
            date,
            year,
            month_name,
            weekday_name,
            hour,
            season_name,
            weather_name,
            total_rentals
        FROM hourly_rentals
        ORDER BY total_rentals DESC
        LIMIT 10;
    """
}


# -----------------------------
# 5. Run queries and save outputs
# -----------------------------
for name, query in queries.items():
    result = pd.read_sql(query, conn)

    print(f"\n{name}:")
    print(result.head(10))

    output_file = output_tables_path / f"{name}.csv"
    result.to_csv(output_file, index=False)

    print(f"Saved to: {output_file}")


# -----------------------------
# 6. Close database connection
# -----------------------------
conn.close()

print("\nSQL analysis complete.")
print(f"Database saved as: {database_path}")
print(f"Query outputs saved in: {output_tables_path}")