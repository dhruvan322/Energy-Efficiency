# train_model.py
"""
Train multiple regressors and classifiers for the ENB2012 / Energy Efficiency dataset.
Saves models to backend/model/, scaler, and metrics.json for later display.
"""
import os, joblib, json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier, RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, classification_report
from datetime import datetime

MODEL_DIR = os.path.join('backend', 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------- Load dataset --------------------
df = pd.read_csv(r'data\energy_efficiency_data.csv')

# rename columns if necessary
if 'Y1' in df.columns and 'Y2' in df.columns:
    df.rename(columns={
        'X1': 'Relative_Compactness',
        'X2': 'Surface_Area',
        'X3': 'Wall_Area',
        'X4': 'Roof_Area',
        'X5': 'Overall_Height',
        'X6': 'Orientation',
        'X7': 'Glazing_Area',
        'X8': 'Glazing_Area_Distribution',
        'Y1': 'Heating_Load',
        'Y2': 'Cooling_Load'
    }, inplace=True)

print("Columns:", df.columns.tolist())

# -------------------- Prepare features and targets --------------------
FEATURE_COLS = [c for c in df.columns if c not in ('Heating_Load', 'Cooling_Load')]
X = df[FEATURE_COLS].astype(float).values
y_heat = df['Heating_Load'].values
y_cool = df['Cooling_Load'].values

# -------------------- Create classification targets by binning (3 classes) --------------------
# qcut ensures roughly equal distribution among classes
heat_bins = pd.qcut(df['Heating_Load'], q=3, labels=['Low', 'Medium', 'High'])
cool_bins = pd.qcut(df['Cooling_Load'], q=3, labels=['Low', 'Medium', 'High'])

# Save class labels ordering for later use
class_labels = ['Low', 'Medium', 'High']

# -------------------- Train/test split --------------------
X_train, X_test, yh_train, yh_test, yc_train, yc_test, hcls_train, hcls_test, ccls_train, ccls_test = train_test_split(
    X, y_heat, y_cool, heat_bins, cool_bins, test_size=0.2, random_state=42
)

# -------------------- Scaling --------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

# -------------------- Model lists --------------------
regressors = {
    'LinearRegression': LinearRegression(),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42, n_estimators=50),
    'RandomForest': RandomForestRegressor(random_state=42, n_estimators=100),
    'GradientBoost': GradientBoostingRegressor(random_state=42, n_estimators=100)
}

classifiers = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42, n_estimators=50),
    'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
    'GradientBoost': GradientBoostingClassifier(random_state=42, n_estimators=100)
}

metrics = {
    'trained_at': datetime.utcnow().isoformat() + 'Z',
    'regression': {'heating': {}, 'cooling': {}},
    'classification': {'heating': {}, 'cooling': {}},
    'class_labels': class_labels
}

# -------------------- Train regressors for Heating & Cooling --------------------
for name, model in regressors.items():
    # heating
    m_h = model
    m_h.fit(X_train_scaled, yh_train)
    yh_pred = m_h.predict(X_test_scaled)
    metrics['regression']['heating'][name] = {
        'r2': float(r2_score(yh_test, yh_pred)),
        'mae': float(mean_absolute_error(yh_test, yh_pred)),
        'mse': float(mean_squared_error(yh_test, yh_pred))
    }
    joblib.dump(m_h, os.path.join(MODEL_DIR, f'model_heating_reg_{name}.pkl'))

    # cooling (train separate instance of same model type)
    # create a fresh instance to avoid state-sharing
    m_c = type(model)() if name not in ('KNN','AdaBoost','RandomForest','GradientBoost') else model.__class__(**getattr(model,'get_params',{})())
    # but simpler: instantiate same class with similar params from regressors dict
    m_c = regressors[name].__class__(**regressors[name].get_params()) if hasattr(regressors[name], 'get_params') else regressors[name]
    m_c.fit(X_train_scaled, yc_train)
    yc_pred = m_c.predict(X_test_scaled)
    metrics['regression']['cooling'][name] = {
        'r2': float(r2_score(yc_test, yc_pred)),
        'mae': float(mean_absolute_error(yc_test, yc_pred)),
        'mse': float(mean_squared_error(yc_test, yc_pred))
    }
    joblib.dump(m_c, os.path.join(MODEL_DIR, f'model_cooling_reg_{name}.pkl'))

# -------------------- Train classifiers for Heating & Cooling --------------------
for name, clf in classifiers.items():
    # heating classifier
    clf_h = clf.__class__(**clf.get_params()) if hasattr(clf, 'get_params') else clf
    clf_h.fit(X_train_scaled, hcls_train)
    h_pred = clf_h.predict(X_test_scaled)
    metrics['classification']['heating'][name] = {
        'accuracy': float(accuracy_score(hcls_test, h_pred)),
        'report': classification_report(hcls_test, h_pred, output_dict=True)
    }
    joblib.dump(clf_h, os.path.join(MODEL_DIR, f'model_heating_clf_{name}.pkl'))

    # cooling classifier
    clf_c = clf.__class__(**clf.get_params()) if hasattr(clf, 'get_params') else clf
    clf_c.fit(X_train_scaled, ccls_train)
    c_pred = clf_c.predict(X_test_scaled)
    metrics['classification']['cooling'][name] = {
        'accuracy': float(accuracy_score(ccls_test, c_pred)),
        'report': classification_report(ccls_test, c_pred, output_dict=True)
    }
    joblib.dump(clf_c, os.path.join(MODEL_DIR, f'model_cooling_clf_{name}.pkl'))

# -------------------- Save metrics --------------------
with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2)

print("✅ Training complete. Models & metrics saved to", MODEL_DIR)
