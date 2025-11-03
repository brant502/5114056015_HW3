import pandas as pd

# --- 1. Data Understanding ---

# Load the dataset
file_path = 'realestate.csv'
df = pd.read_csv(file_path)

# Display the first 5 rows of the dataframe
print("--- First 5 Rows ---")
print(df.head())
print("\n" + "="*50 + "\n")

# Display summary information about the dataframe
print("--- Dataframe Info ---")
df.info()
print("\n" + "="*50 + "\n")

# Display descriptive statistics
print("--- Descriptive Statistics ---")
print(df.describe())


# --- 2. Data Preparation & Feature Analysis ---

# Drop the 'No' column as it is just an index
df = df.drop('No', axis=1)

# Define features (X) and target (y)
features = ['TransactionDate', 'HouseAge', 'DistanceToMRT', 'NumberConvenienceStores', 'Latitude', 'Longitude']
target = 'PriceOfUnitArea'

X = df[features]
y = df[target]

# --- Feature Correlation Analysis ---
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate correlation matrix
corr_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Real Estate Data')

# Save the heatmap as an image
heatmap_path = 'correlation_heatmap.png'
plt.savefig(heatmap_path)
print(f"\n--- Correlation heatmap saved to {heatmap_path} ---\n")


# --- Data Splitting ---
from sklearn.model_selection import train_test_split

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Data Splitting Summary ---")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


# --- 3. Modeling ---
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import statsmodels.api as sm

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)


# --- 4. Evaluation ---

print("\n" + "="*50 + "\n")
print("--- Model Evaluation ---")

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# For a more detailed statistical summary and prediction intervals, we use statsmodels
# Add a constant to the predictor variables for the intercept
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Fit the OLS (Ordinary Least Squares) model
sm_model = sm.OLS(y_train, X_train_sm).fit()

# Get prediction results
predictions = sm_model.get_prediction(X_test_sm)
pred_summary = predictions.summary_frame(alpha=0.05) # alpha=0.05 for 95% confidence

# Extract prediction intervals
y_pred_sm = pred_summary['mean']
pred_ci_lower = pred_summary['obs_ci_lower']
pred_ci_upper = pred_summary['obs_ci_upper']

print("\n--- Statsmodels OLS Results ---")
print(sm_model.summary())


# --- 5. Deployment (Visualization) ---

print("\n" + "="*50 + "\n")
print("--- Generating Prediction Plot ---")

# Create a dataframe for plotting
plot_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_sm, 'Lower_PI': pred_ci_lower, 'Upper_PI': pred_ci_upper})
plot_df = plot_df.sort_values(by='Actual')

plt.figure(figsize=(12, 8))

# Scatter plot for actual vs predicted values
sns.scatterplot(x='Actual', y='Predicted', data=plot_df, label='Predictions', alpha=0.7)

# Prediction interval
plt.fill_between(plot_df['Actual'], plot_df['Lower_PI'], plot_df['Upper_PI'], color='gray', alpha=0.2, label='95% Prediction Interval')

# Line for perfect prediction (y=x)
plt.plot([plot_df['Actual'].min(), plot_df['Actual'].max()], [plot_df['Actual'].min(), plot_df['Actual'].max()], 'r--', label='Perfect Prediction')

plt.title('Actual vs. Predicted Prices with 95% Prediction Interval')
plt.xlabel('Actual Price of Unit Area')
plt.ylabel('Predicted Price of Unit Area')
plt.legend()
plt.grid(True)

# Save the plot
prediction_plot_path = 'prediction_plot.png'
plt.savefig(prediction_plot_path)

print(f"\n--- Prediction plot saved to {prediction_plot_path} ---")