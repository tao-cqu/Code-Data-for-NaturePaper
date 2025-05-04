import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import matplotlib.pyplot as plt

# Load data (replace with your dataset)
data = pd.read_excel('your_data.xlsx')
x = data.iloc[:, :]
y = data.iloc[:, :]

# Data preprocessing: Standardize input features
scaler_X = StandardScaler().fit(x)
X_scaled = scaler_X.transform(x)

# Normalize target variable
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Split data into training and testing sets
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=2024
)

# Initialize Random Forest model with hyperparameters
model = RandomForestRegressor(
    n_estimators=2000,
    max_depth=50,
    min_samples_split=5,
    min_samples_leaf=1
)

# Wrap model for multi-output regression
multioutput_model = MultiOutputRegressor(model)
multioutput_model.fit(X_train_scaled, y_train_scaled)

# Print feature importances
for i, est in enumerate(multioutput_model.estimators_):
    print(f"Feature importances for output {i + 1}: {est.feature_importances_}")


# Define adjusted R? metric
def r2_adjusted(r2, y_true, n_features):
    n = y_true.shape[0]
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)


# Generate predictions
y_train_pred_scaled = multioutput_model.predict(X_train_scaled)
y_test_pred_scaled = multioutput_model.predict(X_test_scaled)

# Set Nature-style plotting parameters
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Arial',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
    'figure.figsize': [6, 6],
    'axes.spines.right': False,
    'axes.spines.top': False,
    'legend.frameon': False
})

# Plot predicted vs. true values
for i in range(y_scaled.shape[1]):
    fig, ax = plt.subplots()

    # Scatter plots
    ax.scatter(y_train_pred_scaled[:, i], y_train_scaled[:, i],
               color='#1f77b4', label='Training', s=60, alpha=0.7)
    ax.scatter(y_test_pred_scaled[:, i], y_test_scaled[:, i],
               color='#d62728', label='Testing', s=60, alpha=0.7)

    # 1:1 line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.8, label='1:1 line')

    # Labels and limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Predicted value (normalized)')
    ax.set_ylabel('True value (normalized)')

    # Add metrics
    r2_train = r2_score(y_train_scaled[:, i], y_train_pred_scaled[:, i])
    r2_test = r2_score(y_test_scaled[:, i], y_test_pred_scaled[:, i])
    r2_adj = r2_adjusted(r2_test, y_test_scaled[:, i], X_scaled.shape[1])

    stats_text = f'$R^2$ (train) = {r2_train:.3f}\n$R^2$ (test) = {r2_test:.3f}\n$R^2$ (adj) = {r2_adj:.3f}'
    ax.text(0.98, 0.15, stats_text, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=11,
            bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

# Calculate final metrics
print("Test MSE:", np.mean((y_test_pred_scaled - y_test_scaled) ** 2))
print("Test R?:", r2_score(y_test_scaled, y_test_pred_scaled))
print("Adjusted R?:", r2_adjusted(r2_score(y_test_scaled, y_test_pred_scaled),
                                  y_test_scaled, X_scaled.shape[1]))


