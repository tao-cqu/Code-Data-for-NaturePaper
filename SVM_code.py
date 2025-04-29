from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load dataset (replace with your file path)
data = pd.read_excel()
x = data.iloc[:, :]
y = data.iloc[:, :]

# Data preprocessing
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(x)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

# Split dataset
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=2024
)

# Initialize and train SVR model
svr_model = SVR(
    kernel='rbf',
    C=0.1,          # Regularization parameter
    gamma=0.1,      # Kernel coefficient
    epsilon=0.1     # Epsilon-tube for no penalty
)
svr_model.fit(X_train_scaled, y_train_scaled)

# Generate predictions
y_pred_test = svr_model.predict(X_test_scaled)
y_pred_train = svr_model.predict(X_train_scaled)

# Calculate performance metrics
def adjusted_r2(r2, n_samples, n_features):
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

test_r2 = r2_score(y_test_scaled, y_pred_test)
train_r2 = r2_score(y_train_scaled, y_pred_train)
adj_r2 = adjusted_r2(test_r2, y_test_scaled.shape[0], X_scaled.shape[1])

print(f'''
Performance Metrics:
- Train R?: {train_r2:.3f}
- Test R?: {test_r2:.3f}
- Adjusted R?: {adj_r2:.3f}
- Test MSE: {mean_squared_error(y_test_scaled, y_pred_test):.3e}
''')


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

# Create prediction vs true value plot
plt.figure()
plt.scatter(y_test_scaled, y_pred_test, color='#d62728', alpha=0.7,
            edgecolor='w', linewidth=0.5)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('True Values (normalized)')
plt.ylabel('Predicted Values (normalized)')
plt.tight_layout()
plt.savefig('svr_predictions.png', dpi=300)
plt.show()