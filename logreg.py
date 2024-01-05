import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('temp_df.csv', index_col=0)

X = data.drop(['win'], axis=1)
y = data['win']

print(X.head(20))

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Chronological split
split_index = int(0.7 * len(data))
X_train = data.drop(['win'], axis=1).iloc[:split_index]
y_train = data['win'].iloc[:split_index]
X_test = data.drop(['win'], axis=1).iloc[split_index:]
y_test = data['win'].iloc[split_index:]


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

logistic_regressor = LogisticRegression()
logistic_regressor.fit(X_train, y_train)

y_train_pred = logistic_regressor.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy}")

y_pred = logistic_regressor.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(report)

intercept = logistic_regressor.intercept_[0]
coefficients = logistic_regressor.coef_[0]
print(intercept)
print(coefficients)


feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': coefficients
}).sort_values(by='Importance', ascending=False)
print(feature_importances)

coef_rank = logistic_regressor.coef_[0][0]
coef_opp_rank = logistic_regressor.coef_[0][1]