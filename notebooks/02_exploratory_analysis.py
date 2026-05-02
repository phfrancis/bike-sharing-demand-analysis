from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------
# 1. File paths
# -----------------------------
clean_data_path = Path("data/processed/bike_hour_clean.csv")
figures_path = Path("outputs/figures")
tables_path = Path("outputs/tables")

figures_path.mkdir(parents=True, exist_ok=True)
tables_path.mkdir(parents=True, exist_ok=True)


# -----------------------------
# 2. Load cleaned data
# -----------------------------
df = pd.read_csv(clean_data_path)
df["date"] = pd.to_datetime(df["date"])

print("\nCleaned data loaded:")
print(df.head())
print("\nShape:")
print(df.shape)


# -----------------------------
# 3. Plot: Average rentals by hour
# -----------------------------
hourly_avg = (
    df.groupby("hour", as_index=False)["total_rentals"]
      .mean()
)

hourly_avg.to_csv(tables_path / "hourly_avg_from_python.csv", index=False)

plt.figure(figsize=(10, 5))
plt.plot(hourly_avg["hour"], hourly_avg["total_rentals"], marker="o")
plt.xlabel("Hour of Day")
plt.ylabel("Average Rentals")
plt.title("Average Bike Rentals by Hour")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig(figures_path / "average_rentals_by_hour.png", dpi=300)
plt.show()


# -----------------------------
# 4. Plot: Average rentals by season
# -----------------------------
season_order = ["spring", "summer", "fall", "winter"]

season_label_map = {
    "spring": "Spring",
    "summer": "Summer",
    "fall": "Fall",
    "winter": "Winter"
}

season_avg = (
    df.groupby("season_name", as_index=False)["total_rentals"]
      .mean()
)

season_avg["season_name"] = pd.Categorical(
    season_avg["season_name"],
    categories=season_order,
    ordered=True
)

season_avg = season_avg.sort_values("season_name")
season_avg["season_label"] = season_avg["season_name"].map(season_label_map)

season_avg.to_csv(tables_path / "season_avg_from_python.csv", index=False)

plt.figure(figsize=(8, 5))
plt.bar(season_avg["season_label"], season_avg["total_rentals"])
plt.xlabel("Season")
plt.ylabel("Average Rentals")
plt.title("Average Bike Rentals by Season")
plt.tight_layout()
plt.savefig(figures_path / "average_rentals_by_season.png", dpi=300)
plt.show()


# -----------------------------
# 5. Plot: Average rentals by weather condition
# -----------------------------
weather_label_map = {
    "clear_or_partly_cloudy": "Clear / Partly Cloudy",
    "mist_or_cloudy": "Mist / Cloudy",
    "light_rain_or_snow": "Light Rain / Snow",
    "heavy_rain_or_snow": "Heavy Rain / Snow"
}

weather_order = [
    "Clear / Partly Cloudy",
    "Mist / Cloudy",
    "Light Rain / Snow",
    "Heavy Rain / Snow"
]

weather_avg = (
    df.groupby("weather_name", as_index=False)["total_rentals"]
      .mean()
)

weather_avg["weather_label"] = weather_avg["weather_name"].map(weather_label_map)

weather_avg["weather_label"] = pd.Categorical(
    weather_avg["weather_label"],
    categories=weather_order,
    ordered=True
)

weather_avg = weather_avg.sort_values("weather_label")

weather_avg.to_csv(tables_path / "weather_avg_from_python.csv", index=False)

plt.figure(figsize=(10, 5))
plt.bar(weather_avg["weather_label"], weather_avg["total_rentals"])
plt.xlabel("Weather Condition")
plt.ylabel("Average Rentals")
plt.title("Average Bike Rentals by Weather Condition")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(figures_path / "average_rentals_by_weather.png", dpi=300)
plt.show()


# -----------------------------
# 6. Plot: Temperature vs rentals
# -----------------------------
plt.figure(figsize=(8, 5))
plt.scatter(df["temp"], df["total_rentals"], alpha=0.3)
plt.xlabel("Normalized Temperature")
plt.ylabel("Total Rentals")
plt.title("Bike Rentals vs Temperature")
plt.tight_layout()
plt.savefig(figures_path / "rentals_vs_temperature.png", dpi=300)
plt.show()


# -----------------------------
# 7. Plot: Monthly rental trend
# -----------------------------
monthly_total = (
    df.groupby(["year", "month"], as_index=False)["total_rentals"]
      .sum()
)

monthly_total["year_month"] = (
    monthly_total["year"].astype(str)
    + "-"
    + monthly_total["month"].astype(str).str.zfill(2)
)

monthly_total.to_csv(tables_path / "monthly_total_from_python.csv", index=False)

plt.figure(figsize=(12, 5))
plt.plot(monthly_total["year_month"], monthly_total["total_rentals"], marker="o")
plt.xlabel("Month")
plt.ylabel("Total Rentals")
plt.title("Monthly Bike Rental Trend")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(figures_path / "monthly_rental_trend.png", dpi=300)
plt.show()


# -----------------------------
# 8. Plot: Working day vs non-working day
# -----------------------------
workingday_avg = (
    df.groupby("workingday", as_index=False)["total_rentals"]
      .mean()
)

workingday_avg["day_type"] = workingday_avg["workingday"].map({
    0: "non_working_day",
    1: "working_day"
})

workingday_avg["day_type_label"] = workingday_avg["workingday"].map({
    0: "Non-Working Day",
    1: "Working Day"
})

workingday_avg.to_csv(tables_path / "workingday_avg_from_python.csv", index=False)

plt.figure(figsize=(7, 5))
plt.bar(workingday_avg["day_type_label"], workingday_avg["total_rentals"])
plt.xlabel("Day Type")
plt.ylabel("Average Rentals")
plt.title("Average Bike Rentals: Working Day vs Non-Working Day")
plt.tight_layout()
plt.savefig(figures_path / "workingday_vs_nonworkingday.png", dpi=300)
plt.show()


# -----------------------------
# 9. Plot: Average rentals by hour and weekday
# -----------------------------
weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

df["weekday_name"] = pd.Categorical(
    df["weekday_name"],
    categories=weekday_order,
    ordered=True
)

hour_weekday_avg = (
    df.groupby(["hour", "weekday_name"], observed=False)["total_rentals"]
      .mean()
      .reset_index()
)

hour_weekday_avg.to_csv(tables_path / "hour_weekday_avg_from_python.csv", index=False)

pivot = hour_weekday_avg.pivot(
    index="hour",
    columns="weekday_name",
    values="total_rentals"
)

plt.figure(figsize=(10, 6))
plt.imshow(pivot, aspect="auto")
plt.colorbar(label="Average Rentals")
plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
plt.yticks(range(24), range(24))
plt.title("Average Bike Rentals by Hour and Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Hour of Day")
plt.tight_layout()
plt.savefig(figures_path / "heatmap_hour_vs_weekday.png", dpi=300)
plt.show()


# -----------------------------
# 10. Save summary table
# -----------------------------
summary = df["total_rentals"].describe()

print("\nSummary statistics for total rentals:")
print(summary)

summary.to_csv(tables_path / "total_rentals_summary.csv")


# -----------------------------
# 11. Print key insights
# -----------------------------
print("\nKey Insights:")
print("- Peak demand occurs during commute hours, especially around 8 AM and 5–6 PM.")
print("- Fall has the highest average rentals, while spring has the lowest.")
print("- Clear or partly cloudy weather is associated with the highest average demand.")
print("- Rentals generally increase with temperature, though the relationship is noisy.")
print("- Working days show slightly higher average rentals than non-working days.")
print("- Monthly totals show both seasonality and higher demand in 2012 than 2011.")

print("\nExploratory figures saved to:")
print(figures_path)

print("\nExploratory tables saved to:")
print(tables_path)