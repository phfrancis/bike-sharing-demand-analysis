# 🚲 Bike Sharing Demand Analysis

## 📌 Project Overview

This project analyzes and predicts bike rental demand using an end-to-end data workflow that combines:

- **Python (Pandas, NumPy, Matplotlib)**
- **SQL (SQLite)**
- **Machine Learning (Scikit-learn)**
- **Business Intelligence (Power BI)**

The objective is to uncover demand patterns and build predictive models using historical bike-sharing data, with a strong focus on **time-based feature engineering** and **interpretability**.

---

## 📊 Key Components

### 🔹 1. Data Cleaning
- Loaded and cleaned raw hourly bike rental data
- Renamed columns for clarity and consistency
- Created categorical labels:
  - Season
  - Weather condition
  - Day type (working vs non-working)
- Generated additional date-based features (month, weekday, etc.)

---

### 🔹 2. Exploratory Data Analysis
Analyzed demand patterns across multiple dimensions:

- **Hour of Day** → strong commute peaks (morning & evening)
- **Season** → highest demand in fall, lowest in spring
- **Weather** → highest usage in clear conditions
- **Working vs Non-Working Days** → slightly higher weekday demand

Visual outputs include:
- Line plots  
- Bar charts  
- Heatmaps  
- Scatter plots  

---

### 🔹 3. Machine Learning

#### Models Implemented:
- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- Mean Baseline (for comparison)  

#### Feature Engineering:
- Lag features:
  - Previous hour  
  - Same hour previous day  
  - Same hour previous week  
- Rolling statistics:
  - 24-hour average  
  - 7-day average  
  - 7-day variability  

#### Evaluation Metrics:
- RMSE (Root Mean Squared Error)  
- MAE (Mean Absolute Error)  
- R² (Coefficient of Determination)  

---

### 🔹 4. SQL Analysis
- Built a **SQLite database** from cleaned data  
- Wrote analytical queries for:
  - Rentals by hour  
  - Rentals by season  
  - Rentals by weather  
  - Working vs non-working demand  
  - Monthly trends  
  - Top 10 highest demand periods  

---

### 🔹 5. Power BI Dashboard

Interactive dashboard includes:

- **Demand Overview**  
- **Monthly Trends**  
- **Model Performance Comparison**  
- **Feature Importance Analysis**  
- **Drillthrough Page (Season-Level Deep Dive)**  

---

## 📈 Key Insights

- Bike rental demand is **highly time-dependent**  
- **Lag features dominate** both correlation and model importance  
- Weather impacts demand but plays a **secondary role**  
- Linear Regression achieved the **best performance** after feature engineering  
- Demand follows strong **daily and weekly cycles**  

---

## 🛠️ Tech Stack

- **Python**: Pandas, NumPy, Matplotlib, Scikit-learn  
- **SQL**: SQLite  
- **BI Tool**: Power BI  

---

## 📂 Project Structure

```
bike-sharing-demand-analysis/

├── data/
│   ├── raw/
│   └── processed/

├── notebooks/
│   ├── 01_data_cleaning.py
│   ├── 02_exploratory_analysis.py
│   ├── 03_ml_modeling.py
│   └── 04_sql_analysis.py

├── outputs/
│   ├── figures/
│   └── tables/

├── powerbi/
│   └── bike_sharing_dashboard.pbix

├── sql/
│   └── bike_sharing.db

├── README.md
├── requirements.txt
```

---

## ⚙️ How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the pipeline

```bash
python notebooks/01_data_cleaning.py
python notebooks/02_exploratory_analysis.py
python notebooks/03_ml_modeling.py
python notebooks/04_sql_analysis.py
```

---

## 📊 Dashboard

To explore the interactive dashboard:

```bash
powerbi/bike_sharing_dashboard.pbix
```

---

## 📸 Dashboard Preview

### Demand Overview
![Demand Overview](outputs/figures/average_rentals_by_hour.png)

### Monthly Trends
![Monthly Trends](outputs/figures/monthly_rental_trend.png)

### Model Performance
![Model Performance](outputs/figures/ml_actual_vs_predicted_monthly.png)

### Feature Importance
![Feature Importance](outputs/figures/random_forest_feature_importance.png)

---

## 🚀 Key Takeaway

This project demonstrates that **feature engineering—especially time-based features—is more impactful than model complexity** in predicting bike rental demand.
