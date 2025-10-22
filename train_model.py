import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import os

# Set up the correct file path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'data', 'energy_efficiency_data.csv')

# Load data
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Data file not found at {data_path}. Please ensure the file exists.")

# Prepare features and target
X = df.iloc[:, :8]  # First 8 columns are features
y = df.iloc[:, 8]   # 9th column is heating load

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate R² score
r2 = r2_score(y_test, y_pred)
print(f'R² Score: {r2:.4f}')

# Create output directory if it doesn't exist
output_dir = os.path.join(current_dir, 'models')
os.makedirs(output_dir, exist_ok=True)

# Save the model and scaler
model_path = os.path.join(output_dir, 'heating_load_model.joblib')
scaler_path = os.path.join(output_dir, 'scaler.joblib')
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
print(f"Model saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")