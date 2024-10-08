import os
import joblib
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# Dataset laden
data = pd.read_csv('HR_Analytics.csv')
# Kenmerken en target kiezen
# Encodering van 'Gender' (1 voor 'Male', 0 voor 'Female')
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
# One-hot encoding voor categorische variabelen zoals 'Department' en 'BusinessTravel'
data = pd.get_dummies(
    data, columns=['Department', 'BusinessTravel'], drop_first=True)
# Features instellen
features = ['Age', 'Gender', 'Education', 'TotalWorkingYears', 'MonthlyIncome'] + \
    [col for col in data.columns if 'Department' in col or 'BusinessTravel' in col]
X = data[features]
# Target instellen (verondersteld dat we 'Attrition' willen voorspellen; aanpassen indien nodig)
# Attrition encoderen naar 1 voor 'Yes', 0 voor 'No'
y = data['Attrition'].map({'Yes': 1, 'No': 0})
# Train-test splitsen
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
# Random Forest Model trainen
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Voorspellingen maken en rapport printen
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
# Model opslaan
joblib.dump(model, 'hiring_model.pkl')
