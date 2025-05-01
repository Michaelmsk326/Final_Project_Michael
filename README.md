# 377_Final_Dashboard_RE
## Website: https://finalprojectmichael-jycomappwpcg6yc4wbd8gxs.streamlit.app/

# Chicago Neighborhood Analysis: Crime, Housing Prices & School Ratings

## Project Overview  
This repository analyzes how public‐safety, housing market values, and school quality co‐vary across Chicago ZIP Codes. We merge police‐reported incident counts, Zillow home‐value indices, and state school‐rating data, then train predictive models and visualize geographic and statistical relationships.

---

## Repository Contents  
- **final_merged_dataset.csv**  
  The master ZIP‐level file combining:  
  - `Total_Crimes` (2023 incident count)  
  - `average_housing_cost_2023` (Zillow home‐value index)  
  - `Overall_Rating` (numeric mapping of school performance tiers)  
  - `Low_Income_Percentage` (fraction of students on free/reduced lunch)  
  - `Rating_Status` (–1 for missing ratings, 0/1 for present)  
  - `School_Latitude`, `School_Longitude` (geographic centroids)  

- **project.ipynb**  
  A self‐contained workflow that:  
  1. Loads & preprocesses the data  
  2. Splits into train/test sets  
  3. Trains XGBoost regressors for both housing cost and school rating  
  4. Evaluates models (MSE, RMSE, MAE, R²)  
  5. Plots feature importances, scatter‐plots, geographic maps, and a correlation heatmap  

- **plots**  
  Output directory (created at runtime) containing:  
  - `housing_feature_importance.png`  
  - `rating_feature_importance.png`  
  - `housing_prediction_scatter.png`  
  - `chicago_housing_map.png`  
  - `chicago_school_rating_map.png`  
  - `chicago_crime_map.png`  
  - `housing_crime_scatter.png`  
  - `housing_income_scatter.png`  
  - `chicago_zipcode_housing_schools.png`  
  - `correlation_matrix.png`  
  - `chicago_housing_rating_combined.png

- **packages**

  - pip install \
  pandas numpy matplotlib seaborn xgboost scikit-learn contextily