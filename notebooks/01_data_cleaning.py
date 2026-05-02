import pandas as pd
from pathlib import Path


# -----------------------------
# 1. File paths
# -----------------------------
raw_path = Path("data/raw/hour.csv")
processed_path = Path("data/processed/bike_hour_clean.csv")


# -----------------------------
# 2. Load data
# -----------------------------
df = pd.read_csv(raw_path)

print("\nRaw data preview:")
print(df.head())

print("\nRaw data info:")
print(df.info())


# -----------------------------
# 3. Convert date column
# -----------------------------
df["dteday"] = pd.to_datetime(df["dteday"])


# -----------------------------
# 4. Rename columns for clarity
# -----------------------------
df = df.rename(columns={
    "instant": "record_id",
    "dteday": "date",
    "yr": "year_code",
    "mnth": "month",
    "hr": "hour",
    "weathersit": "weather_code",
    "hum": "humidity",
    "cnt": "total_rentals",
    "casual": "casual_rentals",
    "registered": "registered_rentals"
})


# -----------------------------
# 5. Decode categorical variables
# -----------------------------
season_map = {
    1: "spring",
    2: "summer",
    3: "fall",
    4: "winter"
}

weather_map = {
    1: "clear_or_partly_cloudy",
    2: "mist_or_cloudy",
    3: "light_rain_or_snow",
    4: "heavy_rain_or_snow"
}

year_map = {
    0: 2011,
    1: 2012
}

df["season_name"] = df["season"].map(season_map)
df["weather_name"] = df["weather_code"].map(weather_map)
df["year"] = df["year_code"].map(year_map)


# -----------------------------
# 6. Add useful date features
# -----------------------------
df["day"] = df["date"].dt.day
df["weekday_name"] = df["date"].dt.day_name()
df["month_name"] = df["date"].dt.month_name()


# -----------------------------
# 7. Check missing values
# -----------------------------
print("\nMissing values by column:")
print(df.isna().sum())


# -----------------------------
# 8. Keep columns in clean order
# -----------------------------
clean_columns = [
    "record_id",
    "date",
    "year",
    "month",
    "month_name",
    "day",
    "hour",
    "weekday",
    "weekday_name",
    "workingday",
    "holiday",
    "season",
    "season_name",
    "weather_code",
    "weather_name",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "casual_rentals",
    "registered_rentals",
    "total_rentals"
]

df_clean = df[clean_columns]

# -----------------------------
# 8.5 Convert categorical columns
# -----------------------------
categorical_cols = [
    "season_name",
    "weather_name",
    "weekday_name",
    "month_name"
]

for col in categorical_cols:
    df_clean[col] = df_clean[col].astype("category")

# -----------------------------
# 9. Save cleaned data
# -----------------------------
processed_path.parent.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(processed_path, index=False)

print("\nCleaned data preview:")
print(df_clean.head())

print("\nCleaned data saved to:")
print(processed_path)

print("\nCleaned data shape:")
print(df_clean.shape)