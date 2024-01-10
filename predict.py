import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('temp_df.csv', index_col=0)
data['lan'] = data['lan'].astype('category')
data['format'] = data['format'].astype('category')

X = data.drop(['win'], axis=1)
y = data['win']



# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Chronological split
split_index = int(0.7 * len(data))
X_train = data.drop(['win'], axis=1).iloc[:split_index]
y_train = data['win'].iloc[:split_index]
X_test = data.drop(['win'], axis=1).iloc[split_index:]
y_test = data['win'].iloc[split_index:]

feature_names = X_train.columns.tolist()
print(feature_names)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Logistic Regression

from sklearn.linear_model import LogisticRegression
def logistic_regression():
    logistic_regressor = LogisticRegression(C= 0.01, penalty='l2', solver='liblinear')
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


# Random Forest

from sklearn.ensemble import RandomForestClassifier
def random_forest():
    rf_classifier = RandomForestClassifier(n_estimators = 100, max_depth=2)

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
        'n_estimators': [50, 100, 200, 350],      # no trees
        'max_depth': [2, 3, 4, 5, 6, 7],     # max tree depth
        # 'min_samples_split': [2, 4, 6],   # samples required to split internal node
        # 'min_samples_leaf': [1, 2, 4]     # ensures leaf node has enough samples
        }

    rf = RandomForestClassifier()
    rf_grid_search = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=5, verbose=2, n_jobs=-1)

    rf_grid_search.fit(X_train, y_train)

    print("Best Parameters for Random Forest:", rf_grid_search.best_params_)
    print("Best Score for Random Forest:", rf_grid_search.best_score_)


# Support Vector Machine

from sklearn.svm import SVC
def support_vector_machine():
    svm_classifier = SVC(C=1, gamma='scale', kernel='linear')

    svm_classifier.fit(X_train, y_train)

    y_train_pred = svm_classifier.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy}")

    y_pred = svm_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Test Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(confusion)
    print("Classification Report:")
    print(report)

def svm_hyperparameter_tuning():
    svm_param_grid = {
        'C': [0.1, 1, 10],               # Regularization parameter
        'kernel': ['linear', 'rbf'],     # Kernel type
        'gamma': ['scale', 'auto']       # Kernel coefficient for 'rbf'
    }

    svm = SVC()

    svm_grid_search = GridSearchCV(estimator=svm, param_grid=svm_param_grid, cv=5, verbose=2, n_jobs=-1)

    svm_grid_search.fit(X_train, y_train)

    print("Best Parameters for SVM:", svm_grid_search.best_params_)
    print("Best Score for SVM:", svm_grid_search.best_score_)


# eXtreme Gradient Boosting

import xgboost as xgb

def xgboost_model():
    params = {'colsample_bytree': 0.9, 'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 300, 'subsample': 0.7}
    xgb_classifier = xgb.XGBClassifier(**params, use_label_encoder=False, 
                                       eval_metric='logloss')

    xgb_classifier.fit(X_train, y_train)

    y_train_pred = xgb_classifier.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy}")

    y_pred = xgb_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Test Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(confusion)
    print("Classification Report:")
    print(report)

    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(xgb_classifier, importance_type='gain', ax=ax, title='Feature Importance', xlabel='F score')

    # Adjusting feature names in the plot
    feature_importances = xgb_classifier.get_booster().get_score(importance_type='gain')
    # Match feature names with their importance scores
    sorted_features = [feature_names[int(f[1:])] for f in sorted(feature_importances, key=lambda x: feature_importances[x])]
    ax.set_yticklabels(sorted_features, rotation='horizontal')

    plt.savefig('xgboost_var_imp.png')

def xgboost_hyperparameter_tuning():
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],    # Number of gradient boosted trees
        'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage used in update
        'max_depth': [3, 4, 5],             # Maximum depth of a tree
        'subsample': [0.7, 0.8, 0.9],       # Subsample ratio of the training instance
        'colsample_bytree': [0.7, 0.8, 0.9] # Subsample ratio of columns when constructing each tree
    }

    xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    xgb_grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=xgb_param_grid, cv=5, verbose=2, n_jobs=-1)

    xgb_grid_search.fit(X_train, y_train)

    print("Best Parameters for XGBoost:", xgb_grid_search.best_params_)
    print("Best Score for XGBoost:", xgb_grid_search.best_score_)


# Gaussian Naive Bayes
    
from sklearn.naive_bayes import GaussianNB

def naive_bayes_model():
    nb_classifier = GaussianNB()

    nb_classifier.fit(X_train, y_train)

    y_train_pred = nb_classifier.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy}")

    y_pred = nb_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Test Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(confusion)
    print("Classification Report:")
    print(report)

# Multi-Layer Perceptron
from sklearn.neural_network import MLPClassifier

def neural_network_model():
    params = {'activation': 'tanh', 'hidden_layer_sizes': (100,100), 'learning_rate_init': 0.001, 'solver': 'sgd'}
    mlp_classifier = MLPClassifier(**params, 
                                   random_state=42)

    mlp_classifier.fit(X_train, y_train)

    y_train_pred = mlp_classifier.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy}")

    y_pred = mlp_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Test Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(confusion)
    print("Classification Report:")
    print(report)

def neural_network_hyperparameter_tuning():
    nn_param_grid = {
        'hidden_layer_sizes': [(50,),(100,), (200,), (50,50), (100,100)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'learning_rate_init': [0.0001, 0.001, 0.01]
    }

    nn_param_grid = {
        'hidden_layer_sizes': [(100,100),(200,200),(200,100),(100, 100, 100), (100, 200, 100), (100, 200, 50)],
        'activation': ['tanh'],
        'solver': ['sgd'],
        'learning_rate_init': [0.001]
    }

    mlp = MLPClassifier(max_iter=10000, random_state=42)

    nn_grid_search = GridSearchCV(estimator=mlp, param_grid=nn_param_grid, cv=5, verbose=2, n_jobs=-1)

    nn_grid_search.fit(X_train, y_train)

    print("Best Parameters for Neural Network:", nn_grid_search.best_params_)
    print("Best Score for Neural Network:", nn_grid_search.best_score_)

if __name__ == "__main__":
    neural_network_model()