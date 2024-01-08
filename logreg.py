import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('temp_df.csv', index_col=0)

X = data.drop(['win'], axis=1)
y = data['win']

# print(X.head(20))

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


from sklearn.linear_model import LogisticRegression
def logistic_regression():
    logistic_regressor = LogisticRegression(C= 0.01, penalty='l1', solver='saga')
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

def logistic_regression_hyperparameter_tuning():
    # Define the parameter grid for Logistic Regression
    lr_param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10],     # inverse of regularization strength
        'penalty': ['l2', 'l1'],            # norm used in regularization
        'solver': ['liblinear', 'saga']     # alg for optimization
    }
    lr = LogisticRegression()
    lr_grid_search = GridSearchCV(estimator=lr, param_grid=lr_param_grid, cv=5, verbose=2, n_jobs=-1)

    lr_grid_search.fit(X_train, y_train)

    print("Best Parameters for Logistic Regression:", lr_grid_search.best_params_)
    print("Best Score for Logistic Regression:", lr_grid_search.best_score_)

from sklearn.ensemble import RandomForestClassifier

def random_forest():
    rf_classifier = RandomForestClassifier(n_estimators = 20, max_depth=4)

    rf_classifier.fit(X_train, y_train)

    y_train_pred = rf_classifier.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy}")

    y_pred = rf_classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Test Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(confusion)
    print("Classification Report:")
    print(report)

    # Feature importances
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_classifier.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print(feature_importances)

def random_forest_hyperparameter_tuning():
    rf_param_grid = {
        'n_estimators': [10, 20, 100],      # no trees
        'max_depth': [3, 4, 5, 10, 20],     # max tree depth
        # 'min_samples_split': [2, 4, 6],   # samples required to split internal node
        # 'min_samples_leaf': [1, 2, 4]     # ensures leaf node has enough samples
        }

    rf = RandomForestClassifier()
    rf_grid_search = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=5, verbose=2, n_jobs=-1)

    rf_grid_search.fit(X_train, y_train)

    print("Best Parameters for Random Forest:", rf_grid_search.best_params_)
    print("Best Score for Random Forest:", rf_grid_search.best_score_)

if __name__ == "__main__":
    logistic_regression_hyperparameter_tuning()