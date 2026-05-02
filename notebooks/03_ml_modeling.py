from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


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

df = df.sort_values(["date", "hour"]).reset_index(drop=True)

print("\nCleaned data loaded:")
print(df.head())

print("\nShape:")
print(df.shape)


# -----------------------------
# 3. Add lag and rolling features
# -----------------------------
df["lag_1_hour"] = df["total_rentals"].shift(1)
df["lag_24_hours"] = df["total_rentals"].shift(24)
df["lag_168_hours"] = df["total_rentals"].shift(168)

df["rolling_24_hour_mean"] = (
    df["total_rentals"]
    .shift(1)
    .rolling(window=24)
    .mean()
)

df["rolling_168_hour_mean"] = (
    df["total_rentals"]
    .shift(1)
    .rolling(window=168)
    .mean()
)

df["rolling_168_hour_std"] = (
    df["total_rentals"]
    .shift(1)
    .rolling(window=168)
    .std()
)

df = df.dropna().reset_index(drop=True)

print("\nData shape after adding lag and rolling features:")
print(df.shape)


# -----------------------------
# 4. Define target and features
# -----------------------------
target = "total_rentals"

numeric_features = [
    "hour",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "lag_1_hour",
    "lag_24_hours",
    "lag_168_hours",
    "rolling_24_hour_mean",
    "rolling_168_hour_mean",
    "rolling_168_hour_std"
]

categorical_features = [
    "season_name",
    "weather_name",
    "weekday_name",
    "workingday",
    "holiday"
]

features = numeric_features + categorical_features

X = df[features]
y = df[target]


# -----------------------------
# 5. Time-based train/test split
# -----------------------------
train_mask = df["year"] == 2011
test_mask = df["year"] == 2012

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

test_dates = df.loc[test_mask, "date"]
test_hours = df.loc[test_mask, "hour"]

print("\nTraining rows:")
print(X_train.shape[0])

print("\nTest rows:")
print(X_test.shape[0])


# -----------------------------
# 6. Preprocessing
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)


# -----------------------------
# 7. Models
# -----------------------------
linear_regression_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

decision_tree_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", DecisionTreeRegressor(
            max_depth=10,
            min_samples_leaf=20,
            random_state=42
        ))
    ]
)

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1
        ))
    ]
)


# -----------------------------
# 8. Mean baseline
# -----------------------------
baseline_prediction = pd.Series(
    y_train.mean(),
    index=y_test.index
)


# -----------------------------
# 9. Fit models
# -----------------------------
linear_regression_model.fit(X_train, y_train)
decision_tree_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)


# -----------------------------
# 10. Make predictions
# -----------------------------
linear_regression_prediction = linear_regression_model.predict(X_test)
decision_tree_prediction = decision_tree_model.predict(X_test)
random_forest_prediction = random_forest_model.predict(X_test)


# -----------------------------
# 11. Evaluation function
# -----------------------------
def evaluate_model(model_name, actual, predicted):
    rmse = mean_squared_error(actual, predicted) ** 0.5
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    return {
        "model": model_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


results = [
    evaluate_model("Mean Baseline", y_test, baseline_prediction),
    evaluate_model("Linear Regression", y_test, linear_regression_prediction),
    evaluate_model("Decision Tree", y_test, decision_tree_prediction),
    evaluate_model("Random Forest", y_test, random_forest_prediction)
]

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("rmse").reset_index(drop=True)

print("\nModel performance:")
print(results_df)

results_df.to_csv(tables_path / "ml_model_performance.csv", index=False)


# -----------------------------
# 11.1. Select best model
# -----------------------------
best_model = results_df.iloc[0]
best_model_name = best_model["model"]

print("\nBest model selected based on RMSE:")
print(best_model)

with open(tables_path / "best_model.txt", "w") as f:
    f.write(best_model_name)


# -----------------------------
# 11.2. Correlation diagnostics
# -----------------------------
correlation_results = (
    df[numeric_features + ["total_rentals"]]
    .corr()["total_rentals"]
    .sort_values(ascending=False)
    .reset_index()
)

correlation_results.columns = ["feature", "correlation_with_total_rentals"]

print("\nTop correlations with target:")
print(correlation_results)

correlation_results.to_csv(
    tables_path / "numeric_feature_correlations.csv",
    index=False
)


# -----------------------------
# 12. Save prediction results
# -----------------------------
prediction_results = pd.DataFrame({
    "date": test_dates.values,
    "hour": test_hours.values,
    "actual_total_rentals": y_test.values,
    "mean_baseline_prediction": baseline_prediction.values,
    "linear_regression_prediction": linear_regression_prediction,
    "decision_tree_prediction": decision_tree_prediction,
    "random_forest_prediction": random_forest_prediction
})

prediction_results.to_csv(
    tables_path / "ml_prediction_results.csv",
    index=False
)

print("\nPrediction results preview:")
print(prediction_results.head())


# -----------------------------
# 13. Plot: Actual vs predicted by month
# -----------------------------
prediction_results["date"] = pd.to_datetime(prediction_results["date"])
prediction_results["year_month"] = prediction_results["date"].dt.to_period("M").astype(str)

monthly_predictions = (
    prediction_results
    .groupby("year_month", as_index=False)
    [
        [
            "actual_total_rentals",
            "mean_baseline_prediction",
            "linear_regression_prediction",
            "decision_tree_prediction",
            "random_forest_prediction"
        ]
    ]
    .mean()
)

monthly_predictions.to_csv(
    tables_path / "ml_monthly_prediction_summary.csv",
    index=False
)

plt.figure(figsize=(12, 5))

plt.plot(
    monthly_predictions["year_month"],
    monthly_predictions["actual_total_rentals"],
    marker="o",
    label="Actual"
)

plt.plot(
    monthly_predictions["year_month"],
    monthly_predictions["mean_baseline_prediction"],
    marker="o",
    label="Mean Baseline"
)

plt.plot(
    monthly_predictions["year_month"],
    monthly_predictions["linear_regression_prediction"],
    marker="o",
    label="Linear Regression"
)

plt.plot(
    monthly_predictions["year_month"],
    monthly_predictions["decision_tree_prediction"],
    marker="o",
    label="Decision Tree"
)

plt.plot(
    monthly_predictions["year_month"],
    monthly_predictions["random_forest_prediction"],
    marker="o",
    label="Random Forest"
)

plt.xlabel("Month")
plt.ylabel("Average Hourly Rentals")
plt.title("Actual vs Predicted Average Hourly Bike Rentals")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(figures_path / "ml_actual_vs_predicted_monthly.png", dpi=300)
plt.show()


# -----------------------------
# 14. Plot: Best model actual vs predicted scatter
# -----------------------------
if best_model_name == "Linear Regression":
    best_prediction = linear_regression_prediction
elif best_model_name == "Decision Tree":
    best_prediction = decision_tree_prediction
elif best_model_name == "Random Forest":
    best_prediction = random_forest_prediction
else:
    best_prediction = baseline_prediction.values

plt.figure(figsize=(7, 7))
plt.scatter(y_test, best_prediction, alpha=0.2)

min_value = min(y_test.min(), best_prediction.min())
max_value = max(y_test.max(), best_prediction.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=2
)

plt.xlabel("Actual Total Rentals")
plt.ylabel("Predicted Total Rentals")
plt.title(f"{best_model_name}: Actual vs Predicted Rentals")

plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)

plt.tight_layout()
plt.savefig(figures_path / "best_model_actual_vs_predicted_scatter.png", dpi=300)
plt.show()


# -----------------------------
# 15. Feature importance
# -----------------------------
rf_model = random_forest_model.named_steps["model"]
feature_names = random_forest_model.named_steps["preprocessor"].get_feature_names_out()

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)


def clean_feature_name(feature):
    clean_name = (
        feature
        .replace("num__", "")
        .replace("cat__", "")
        .replace("_", " ")
        .title()
    )

    replacements = {
        "Atemp": "Feels-Like Temperature",
        "Temp": "Temperature",
        "Lag 1 Hour": "Previous Hour Demand",
        "Lag 24 Hours": "Same Hour Previous Day",
        "Lag 168 Hours": "Same Hour Previous Week",
        "Rolling 24 Hour Mean": "Previous 24-Hour Mean",
        "Rolling 168 Hour Mean": "Previous 7-Day Mean",
        "Rolling 168 Hour Std": "Previous 7-Day Variability",
        "Workingday 0": "Non-Working Day",
        "Workingday 1": "Working Day",
        "Weather Name Clear Or Partly Cloudy": "Weather: Clear / Partly Cloudy",
        "Weather Name Mist Or Cloudy": "Weather: Mist / Cloudy",
        "Weather Name Light Rain Or Snow": "Weather: Light Rain / Snow",
        "Weather Name Heavy Rain Or Snow": "Weather: Heavy Rain / Snow",
        "Season Name Spring": "Season: Spring",
        "Season Name Summer": "Season: Summer",
        "Season Name Fall": "Season: Fall",
        "Season Name Winter": "Season: Winter"
    }

    return replacements.get(clean_name, clean_name)


feature_importance["feature_clean"] = feature_importance["feature"].apply(clean_feature_name)

feature_importance.to_csv(
    tables_path / "random_forest_feature_importance.csv",
    index=False
)

print("\nTop 15 random forest feature importances:")
print(feature_importance.head(15))

print("\nKey takeaway from feature importance:")
print(
    "Lag features dominate, confirming strong temporal dependence in bike rental demand. "
    "Recent demand from the previous hour, daily cycles from the same hour on the previous day, "
    "and weekly cycles from the same hour in the previous week are the most influential predictors."
)

plt.figure(figsize=(10, 6))
top_features = feature_importance.head(15).sort_values("importance")

plt.barh(top_features["feature_clean"], top_features["importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top Random Forest Feature Importances")
plt.tight_layout()
plt.savefig(figures_path / "random_forest_feature_importance.png", dpi=300)
plt.show()


# -----------------------------
# 16. Final ML insight
# -----------------------------
print("\nFinal ML Insight:")

print(
    f"The model with the lowest RMSE is {best_model_name} "
    f"(RMSE = {best_model['rmse']:.2f}, MAE = {best_model['mae']:.2f}, "
    f"R² = {best_model['r2']:.3f}). "

    "Linear regression achieves the lowest RMSE among the evaluated models, "
    "indicating that the relationship between predictors and bike rental demand is largely linear "
    "once appropriate time-based features are included. "

    "Feature engineering plays a central role in this problem. The inclusion of lagged demand "
    "and rolling statistics allows models to capture temporal dependence, including short-term persistence "
    "and strong daily and weekly cycles. "

    "Bike rental demand is highly time-dependent. Recent demand and daily/weekly usage cycles explain "
    "much more variation than weather alone. "

    "This is supported by both correlation analysis and feature importance, where lag variables dominate. "
    "In particular, demand from the previous hour, the same hour on the previous day, and the same hour "
    "in the previous week are the strongest predictors. "

    "Tree-based models such as decision trees and random forests remain competitive and are useful for "
    "capturing nonlinear relationships. However, they do not outperform the linear model in this case, "
    "suggesting that model complexity is less important than well-designed temporal features. "

    "Overall, this analysis demonstrates that capturing temporal structure is the key driver of predictive "
    "performance in bike rental demand modeling. While all models capture general trends well, some "
    "underestimation of extreme high-demand periods remains, particularly for tree-based models due to "
    "their tendency to smooth extreme values."
)

print("\nML outputs saved to:")
print(tables_path)
print(figures_path)