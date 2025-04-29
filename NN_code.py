import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import regularizers
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load dataset (replace with your data path)
data = pd.read_excel()
x = data.iloc[:, :]
y = data.iloc[:, :]

# Data preprocessing: Standardize inputs
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(x)

# Normalize target variable
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Split into training and testing sets
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=2024
)


# Define neural network architecture
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu',
                              kernel_regularizer=regularizers.l2(0.0001),
                              input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu',
                              kernel_regularizer=regularizers.l2(0.0001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(32, activation='relu',
                              kernel_regularizer=regularizers.l2(0.0001)),
        tf.keras.layers.Dense(y_scaled.shape[1])
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

    model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mape']  # Mean Absolute Percentage Error
    )
    return model


# Initialize and train model
model = build_model()
model.summary()

history = model.fit(
    X_train_scaled, y_train_scaled,
    batch_size=512,
    epochs=600,
    validation_split=0.1,
    verbose=0
)

# Evaluate on test set
test_loss = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
print(f'Test loss (MSE): {test_loss[0]:.4f}, Test MAPE: {test_loss[1]:.4f}')


# Plot training history
def plot_training_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(hist['epoch'], hist['loss'], label='Train')
    plt.plot(hist['epoch'], hist['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()

    # Plot MAPE
    plt.subplot(1, 2, 2)
    plt.plot(hist['epoch'], hist['mape'], label='Train')
    plt.plot(hist['epoch'], hist['val_mape'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('MAPE (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()


plot_training_history(history)


# Calculate performance metrics
def adjusted_r2(r2, n_samples, n_features):
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)


y_pred_test = model.predict(X_test_scaled)
y_pred_train = model.predict(X_train_scaled)

# Calculate metrics using sklearn functions
from sklearn.metrics import r2_score, mean_squared_error

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

# Plot predictions vs true values
plt.figure()
plt.scatter(y_test_scaled, y_pred_test, color='#d62728', alpha=0.6,
            edgecolor='w', linewidth=0.5)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('True Values (normalized)')
plt.ylabel('Predictions (normalized)')
plt.tight_layout()
plt.show()