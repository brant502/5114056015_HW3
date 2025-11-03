
# HW2: Multiple Linear Regression for Real Estate Price Prediction

This log summarizes the interaction for the HW2 project, which involved performing a full CRISP-DM analysis to predict real estate prices in Taiwan.

## 1. Project Setup & Data Acquisition
- **Goal:** Complete a university assignment on Multiple Linear Regression.
- **Dataset:** The user selected the [Taiwan Real Estate Prices and Features](https://www.kaggle.com/datasets/noir1112/taiwan-real-estate-prices-and-features-dataset) dataset from Kaggle.
- **Action:**
    - Guided the user to download the dataset.
    - Unzipped the downloaded `archive.zip` file using the `tar` command after `unzip` failed.
    - Identified the data file as `realestate.csv`.

## 2. CRISP-DM: Data Understanding & Preparation
- **Action:** Wrote and executed a Python script (`app.py`) to perform initial data analysis.
- **Findings:** The dataset was clean with no missing values. The `No` column was identified as an unnecessary index.
- **Feature Analysis:**
    - Added `matplotlib`, `seaborn`, and `statsmodels` to `requirements.txt` and installed them.
    - Generated a `correlation_heatmap.png` to analyze feature relationships with the target variable (`PriceOfUnitArea`).
    - Key correlations were identified (e.g., strong negative with `DistanceToMRT`, moderate positive with `NumberConvenienceStores`).
- **Data Split:** The data was split into 80% training and 20% testing sets.

## 3. CRISP-DM: Modeling & Evaluation
- **Model:** A multiple linear regression model was implemented using both `scikit-learn` (for standard metrics) and `statsmodels` (for detailed analysis and prediction intervals).
- **Evaluation:**
    - Calculated and displayed key metrics: MAE, MSE, RMSE, and R-squared (0.68).
    - Generated and displayed a detailed OLS summary from `statsmodels`, which was used for feature significance analysis (identifying `Longitude` as potentially non-significant).

## 4. CRISP-DM: Deployment (Visualization)
- **Action:** Generated a final prediction plot (`prediction_plot.png`) as required by the assignment.
- **Plot Features:**
    - Scatter plot of Actual vs. Predicted prices.
    - A `y=x` line for reference (perfect prediction).
    - A 95% prediction interval, visualizing the model's uncertainty.

## 5. Final Deliverables
- **`README.md`:** Created a comprehensive README file for the project, detailing the entire process, results, and instructions on how to run the analysis.
- **`gemini_interaction_log.md`:** Updated this log with the summary of the HW2 project activities.
- **`AI協助要求.md`:** Informed the user that I cannot export the full conversation and advised them to copy-paste it manually.