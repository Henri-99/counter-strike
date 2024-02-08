import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import os
import csv

# df_full = pd.read_csv('csv/df_full.csv', index_col=0)
df_full = pd.read_csv('csv/df_full_diff.csv', index_col=0)
# df_full = pd.read_csv('csv/df_15.csv', index_col=0)
# df_full = pd.read_csv('csv/df_bo3.csv', index_col=0)
# df_full = pd.read_csv('csv/df_lan.csv', index_col=0)
data = df_full.drop(['match_id', 'datetime', 'team1_id', 'team2_id','team1', 'team2',  't1_score', 't2_score'], axis=1)

# data = df_full.drop([''])
data['lan'] = data['lan'].astype('category')
data['elim'] = data['elim'].astype('category')
data['format'] = data['format'].astype('category')

# Summary of descriptive statistics
# summary = data.describe()

X = data.drop(['win'], axis=1)
y = data['win']

# Choose k, the number of top features to select. For example, k=10
from sklearn.feature_selection import SelectKBest, SelectPercentile, SequentialFeatureSelector, f_classif
def reduce_features(output):
	selector = SelectKBest(f_classif, k=32)
	# selector = SelectPercentile(f_classif, percentile=20)

	X_new = selector.fit_transform(X, y)

	selected_indices = selector.get_support(indices=True)
	selected_features = X.columns[selected_indices]

	scores = selector.scores_
	feature_names = X.columns
	
	features_scores = zip(feature_names, scores)
	sorted_features_scores = sorted(features_scores, key=lambda x: x[1], reverse=True)[:15]

	if output:
		# Separate names and scores for plotting
		names, scores = zip(*sorted_features_scores)

		# Create bar plot
		plt.figure(figsize=(14, 8))
		plt.barh(names[::-1], scores[::-1], color='#abc9ea', edgecolor='#73879d', linewidth=1)
		plt.ylabel('Features')
		plt.xlabel('F-Values')
		# plt.xticks(rotation=45)
		plt.title('')
		plt.savefig("figures/f-statistik.png", bbox_inches='tight',)

	return data[selected_features]
# X = reduce_features(output=False)

# corr_matrix = pd.concat((y, X), axis = 1).corr()

# # Filter out features with a correlation below the threshold
# threshold = 0.05
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap='coolwarm')
# plt.title("Correlation Matrix with 'win'")
# plt.savefig("figures/corr_plot.png", bbox_inches='tight')

selected_features = ['team1_rank', 'team2_rank', 't1_mu', 't1_sigma', 't2_mu', 't2_sigma', 'ts_win_prob',
       't1_elo', 't2_elo', 'elo_win_prob', 't1_wr', 't2_wr', 'wr_diff', 'map_wr', 'xp_diff',
       'avg_hltv_rating_diff', 'avg_pl_rating_diff', 'avg_pistol_wr_diff']
X = X[selected_features]

# Chronological split
split_index = int(0.8 * len(data))
X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

feature_names = X_train.columns.tolist()

# Separate numerical and categorical features
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['category']).columns

# Standardize the numeric features
# norm = MinMaxScaler().fit(X_train[numerical_features])
norm = StandardScaler().fit(X_train[numerical_features])
X_train_norm = pd.DataFrame(norm.transform(X_train[numerical_features]), columns=numerical_features, index=X_train.index)
X_test_norm = pd.DataFrame(norm.transform(X_test[numerical_features]), columns=numerical_features, index=X_test.index)

# Combine with categorical features
X_train = pd.concat([X_train_norm, X_train[categorical_features]], axis=1)
X_test = pd.concat([X_test_norm, X_test[categorical_features]], axis=1)
X_train = pd.concat([X_train[numerical_features], X_train[categorical_features]], axis=1)
X_test = pd.concat([X_test[numerical_features], X_test[categorical_features]], axis=1)

# PCA
# from sklearn.decomposition import PCA
# pca = PCA(n_components=10)  # choose the number of components
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)

# LDA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# lda = LDA(n_components=1) 
# X_train = lda.fit_transform(X_train, y_train)
# X_test = lda.transform(X_test)
# importance = lda.coef_[0]
# for i, v in enumerate(importance):
#     print('Feature: %0d, Score: %.5f' % (i, v))

def print_stats(y_train, y_train_pred, y_test, y_pred, y_pred_proba):
    train_accuracy = accuracy_score(y_train, y_train_pred)*100
    test_accuracy = accuracy_score(y_test, y_pred)*100
    precision = precision_score(y_test, y_pred)*100
    recall = recall_score(y_test, y_pred)*100
    f1 = f1_score(y_test, y_pred)*100
    roc_auc = roc_auc_score(y_test, y_pred_proba)*100



    print("Training ACC & Test ACC & Precision & Recall & F1 Score & ROC AUC \\\\ \\hline")
    print(f"& {train_accuracy:.1f} & {test_accuracy:.1f} & {precision:.1f} & {recall:.1f} & {f1:.1f} & {roc_auc:.1f} \\\\")

# Logistic Regression
from sklearn.linear_model import LogisticRegression
def logistic_regression():
	# full: C: 0.0001
	# fs: {'C': 0.001, 'penalty': 'l2', 'solver': 'liblinear'}
	# params = {'C': 0.1, 'penalty': 'l1', 'solver': 'saga'}
	# params = {'C': 0.0001, 'penalty': 'l2', 'solver': 'liblinear'}
	params = {'C': 1, 'penalty': 'l2', 'solver': 'liblinear'}
	logistic_regressor = LogisticRegression(**params)
	logistic_regressor.fit(X_train, y_train)
	y_train_pred = logistic_regressor.predict(X_train)
	y_pred = logistic_regressor.predict(X_test)
	y_pred_proba = logistic_regressor.predict_proba(X_test)

	coefficients = logistic_regressor.coef_[0]
	feature_importances = pd.DataFrame({
		'Feature': X.columns,
		'Importance': coefficients,
		'Absolute Importance': abs(coefficients)
	}).sort_values(by='Absolute Importance', ascending=False)[:16] # Sort by absolute value

	plt.figure(figsize=(14, 8))
	plt.barh(feature_importances['Feature'][::-1], feature_importances['Absolute Importance'][::-1], color='#abc9ea', edgecolor='#73879d', linewidth=1)
	plt.ylabel('Features')
	plt.xlabel('Absolute Coefficient')
	plt.title('Logistic Regression Feature Importances')
	plt.savefig("figures/logreg-imp.png", bbox_inches='tight',)

	print_stats(y_train, y_train_pred, y_test, y_pred, y_pred_proba[:,-1])
	return y_pred_proba

def logistic_regression_hyperparameter_tuning():
	lr_param_grid = {
		'C': [0.01, 1, 10],
		'penalty': ['l2', 'l1', 'elasticnet'],
		'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
	}

	lr = LogisticRegression(max_iter=10000)
	lr_grid_search = GridSearchCV(estimator=lr, param_grid=lr_param_grid, cv=4, verbose=2, n_jobs=-1)

	lr_grid_search.fit(X_train, y_train)

	print("Best Parameters for Logistic Regression:", lr_grid_search.best_params_)
	print("Best Score for Logistic Regression:", lr_grid_search.best_score_)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
def random_forest():
	# full: {'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 4, 'n_estimators': 200}
	# fs: {'max_depth': 5, 'max_features': 'log2', 'n_estimators': 300}
	# params = {'criterion': 'gini', 'max_depth': 5, 'max_features': 'log2', 'min_samples_leaf': 4, 'min_samples_split': 4, 'n_estimators': 300}
	params = {'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 4, 'n_estimators': 200}
	rf_classifier = RandomForestClassifier(**params)
	rf_classifier.fit(X_train, y_train)
	y_train_pred = rf_classifier.predict(X_train)
	y_pred = rf_classifier.predict(X_test)
	y_pred_proba = rf_classifier.predict_proba(X_test)

	print_stats(y_train, y_train_pred, y_test, y_pred, y_pred_proba[:,-1])


	# Feature importances
	importances = rf_classifier.feature_importances_
	feature_importances_rf = pd.DataFrame({
		'Feature': X.columns,
		'Importance': importances
	}).sort_values(by='Importance', ascending=False)[:16]  # Get top 16 features
	plt.figure(figsize=(14, 8))
	plt.barh(feature_importances_rf['Feature'][::-1], feature_importances_rf['Importance'][::-1], color='#abc9ea', edgecolor='#73879d', linewidth=1)
	plt.ylabel('Features')
	plt.xlabel('Importance')
	plt.title('Random Forest Feature Importances')
	plt.savefig("figures/rf-imp.png", bbox_inches='tight')

	return y_pred_proba

def random_forest_hyperparameter_tuning():
	rf_param_grid = {
		'n_estimators': [130, 131, 132, 133, 134],            # Number of trees in the forest
		'max_features': ['log2'], #['sqrt', 'log2', 0.2, 0.5], # Number of features to consider at every split
		'max_depth': [8, 9, 10],                # Maximum depth of the tree
		# 'min_samples_split': [2,4],                 # Minimum number of samples required to split a node
		# 'min_samples_leaf': [2,4],                  # Minimum number of samples required at a leaf node
		# 'bootstrap': [True, False],                     # Method for sampling data points (with or without replacement)
		# 'class_weight': [None, 'balanced', 'balanced_subsample'], # Weights associated with classes
		# 'criterion': ['gini', 'entropy'],               # Function to measure the quality of a split
		# 'max_leaf_nodes': [None, 10, 20, 30, 50],       # Maximum number of leaf nodes
		# 'min_impurity_decrease': [0.0, 0.1, 0.01, 0.001] # A node will be split if this split induces a decrease of the impurity
	}
	# {'max_depth': 7, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 350}


	rf = RandomForestClassifier()
	rf_grid_search = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=4, verbose=4, n_jobs=-1)

	rf_grid_search.fit(X_train, y_train)

	print("Best Parameters for Random Forest:", rf_grid_search.best_params_)
	print("Best Score for Random Forest:", rf_grid_search.best_score_)
  
# Support Vector Machine
from sklearn.svm import SVC
def support_vector_machine():
	# full = (C=0.1, gamma='scale', kernel='linear', probability=True)
	# fs = {'C': 0.1, 'gamma': 'scale', 'kernel': 'rbf'}
	params = {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}
	svm_classifier = SVC(**params, probability=True)
	svm_classifier.fit(X_train, y_train)
	y_train_pred = svm_classifier.predict(X_train)
	y_pred = svm_classifier.predict(X_test)
	y_pred_proba = svm_classifier.predict_proba(X_test)

	print_stats(y_train, y_train_pred, y_test, y_pred, y_pred_proba[:,-1])

	coefficients = svm_classifier.coef_[0]

	# Convert to DataFrame
	feature_importances_svm = pd.DataFrame({
		'Feature': X.columns,
		'Coefficient': coefficients,
		'Absolute Coefficient': abs(coefficients)
	}).sort_values(by='Absolute Coefficient', ascending=False)[:16]  # Get top 16 features

	# Plot
	plt.figure(figsize=(14, 8))
	plt.barh(feature_importances_svm['Feature'][::-1], feature_importances_svm['Absolute Coefficient'][::-1], color='#abc9ea', edgecolor='#73879d', linewidth=1)
	plt.ylabel('Features')
	plt.xlabel('Absolute Coefficient')
	plt.title('SVM (Linear Kernel) Feature Importances')
	plt.savefig("figures/svm-imp.png", bbox_inches='tight')

	return y_pred_proba

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

def xgboost_model(sfs = False):
	# params = {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 100, 'subsample': 0.7}
	# xgb_classifier = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss', enable_categorical=True)
	# full params = {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 50, 'subsample': 0.7}
	params = {'colsample_bytree': 1, 'learning_rate': 0.08, 'max_depth': 1, 'n_estimators': 208, 'subsample': 0.7}
	# params = {'learning_rate': 0.025, 'max_depth': 2, 'n_estimators': 200}
	xgb_classifier = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss', enable_categorical=True)

	if sfs:
		sfs = SequentialFeatureSelector(xgb_classifier, n_features_to_select=15, direction="forward")
		sfs.fit(X, y)
		selected_features = sfs.get_support()
		feature_names = X.columns  # Assuming X is a DataFrame
		selected_feature_names = feature_names[selected_features]
		print(selected_feature_names)

	xgb_classifier.fit(X_train, y_train)
	y_train_pred = xgb_classifier.predict(X_train)
	y_pred = xgb_classifier.predict(X_test)
	y_pred_proba = xgb_classifier.predict_proba(X_test)

	print_stats(y_train, y_train_pred, y_test, y_pred, y_pred_proba[:,-1])

	# Plotting
	fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,-1])
	ts_fpr, ts_tpr, ts_thresholds = roc_curve(y_test, X_test['ts_win_prob'])
	auc = roc_auc_score(y_test, y_pred_proba[:,-1])
	plt.figure(figsize=(9, 8))
	plt.plot(fpr, tpr, color='#73879d', lw=2,  label=f'XGBoost ROC curve')
	plt.plot(ts_fpr, ts_tpr, color='red', lw=2, label='TrueSkill ROC curve')
	plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.legend(loc="lower right")
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic Curve')
	# plt.legend(loc="lower right")
	plt.savefig("figures/xgb-auc.png", bbox_inches='tight')



	# importances = xgb_classifier.feature_importances_
	# feature_importances_xgb = pd.DataFrame({
	# 	'Feature': X_train.columns,
	# 	'Importance': importances
	# }).sort_values(by='Importance', ascending=False)[:16]  # Top 16 features
	# plt.figure(figsize=(14, 8))
	# plt.barh(feature_importances_xgb['Feature'][::-1], feature_importances_xgb['Importance'][::-1], color='#abc9ea', edgecolor='#73879d', linewidth=1)
	# plt.ylabel('Features')
	# plt.xlabel('Importance')
	# plt.title('XGBoost Feature Importances')
	# plt.savefig("figures/xgb-imp.png", bbox_inches='tight')



	return y_pred_proba

def xgboost_hyperparameter_tuning():
	xgb_param_grid = {
		'n_estimators': [50, 60, 70, 80, 90],    # Number of gradient boosted trees
		'learning_rate': [0.1],  # Step size shrinkage used in update
		'max_depth': [1,2 ],             # Maximum depth of a tree
		# 'subsample': [0.7, 0.8, 0.9],       # Subsample ratio of the training instance
		# 'colsample_bytree': [0.5, 0.75, 1] # Subsample ratio of columns when constructing each tree
	}
	xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', enable_categorical=True)

	xgb_grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=xgb_param_grid, cv=3, verbose=2, n_jobs=-1)

	xgb_grid_search.fit(X_train, y_train)

	print("Best Parameters for XGBoost:", xgb_grid_search.best_params_)
	print("Best Score for XGBoost:", xgb_grid_search.best_score_)

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

def naive_bayes_model():
	nb_classifier = GaussianNB(var_smoothing = 1e-09)
	nb_classifier.fit(X_train, y_train)
	y_train_pred = nb_classifier.predict(X_train)
	y_pred = nb_classifier.predict(X_test)
	y_pred_proba = nb_classifier.predict_proba(X_test)

	print_stats(y_train, y_train_pred, y_test, y_pred, y_pred_proba[:,-1])
	return y_pred_proba

def naive_bayes_hyperparameter_tuning():
	nb_param_grid = {
		'var_smoothing': [1e-09, 1e-08, 1e-07, 1e-06, 1e-05]  # Variance smoothing parameter
	}

	nb = GaussianNB()

	nb_grid_search = GridSearchCV(estimator=nb, param_grid=nb_param_grid, cv=5, verbose=2, n_jobs=-1)

	nb_grid_search.fit(X_train, y_train)

	print("Best Parameters for Naive Bayes:", nb_grid_search.best_params_)
	print("Best Score for Naive Bayes:", nb_grid_search.best_score_)

# k Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier

def knn_model():
	# full params = {'n_neighbors': 10, 'p': 1, 'weights': 'distance'}
	params = {'n_neighbors': 110, 'p': 1, 'weights': 'uniform'}
	knn_classifier = KNeighborsClassifier(**params)
	knn_classifier.fit(X_train, y_train)

	y_train_pred = knn_classifier.predict(X_train)
	y_pred = knn_classifier.predict(X_test)
	y_pred_proba = knn_classifier.predict_proba(X_test)

	print_stats(y_train, y_train_pred, y_test, y_pred, y_pred_proba[:,-1])

	return y_pred_proba

def knn_hyperparameter_tuning():
	knn_param_grid = {
		'n_neighbors': [105, 110, 120],        # Number of neighbors
		'weights': ['uniform',],  # Weight function used in prediction
		'p': [1]                          # Power parameter for the Minkowski metric
		# 'weights': ['uniform', 'distance'],  # Weight function used in prediction
		# 'p': [1, 2]                          # Power parameter for the Minkowski metric
	}

	knn = KNeighborsClassifier()

	knn_grid_search = GridSearchCV(estimator=knn, param_grid=knn_param_grid, cv=5, verbose=2, n_jobs=-1)

	knn_grid_search.fit(X_train, y_train)

	print("Best Parameters for KNN:", knn_grid_search.best_params_)
	print("Best Score for KNN:", knn_grid_search.best_score_)

# Multi-Layer Perceptron
from sklearn.neural_network import MLPClassifier

def neural_network_model():
	# full params = {'activation': 'tanh', 'hidden_layer_sizes': (100,100), 'learning_rate_init': 0.001, 'solver': 'sgd'}
	# params = {'activation': 'tanh', 'hidden_layer_sizes': (100,), 'learning_rate_init': 0.0001, 'solver': 'adam'}
	# params = {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': (5, 4), 'learning_rate_init': 0.001, 'solver': 'sgd'}	
	params = {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': (10,100,10), 'learning_rate_init': 0.001, 'solver': 'sgd'}
	mlp_classifier = MLPClassifier(**params, max_iter=10000,
								   random_state=42)
	mlp_classifier.fit(X_train, y_train)
	y_train_pred = mlp_classifier.predict(X_train)
	y_pred = mlp_classifier.predict(X_test)
	y_pred_proba = mlp_classifier.predict_proba(X_test)

	print_stats(y_train, y_train_pred, y_test, y_pred, y_pred_proba[:,-1])

	return y_pred_proba

	# while True:
	# 	layers = input("Enter layers > ")
	# 	hidden = tuple([int(x) for x in layers.split(',')])
			
	# 	params = {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': hidden, 'learning_rate_init': 0.001, 'solver': 'sgd'}
	# 	mlp_classifier = MLPClassifier(**params, max_iter=10000,
	# 								random_state=42)
	# 	mlp_classifier.fit(X_train, y_train)
	# 	y_train_pred = mlp_classifier.predict(X_train)
	# 	y_pred = mlp_classifier.predict(X_test)
	# 	y_pred_proba = mlp_classifier.predict_proba(X_test)[:,-1]

	# 	print_stats(y_train, y_train_pred, y_test, y_pred, y_pred_proba)

def neural_network_hyperparameter_tuning():
	nn_param_grid = {
		# 'hidden_layer_sizes': [(12,12,12), (11,11,11),(10,10,10)],
		'hidden_layer_sizes': [(5,4), (6,4), (5,3), (6,3), (5,2), (5,1)],
		'activation': ['tanh'],
		# 'activation': ['tanh', 'relu'],
		'solver': ['sgd'],
		# 'solver': ['sgd', 'adam'],
		'learning_rate_init': [0.001],
		# 'learning_rate_init': [0.002, 0.001],
		'alpha' : [0.001]
	}

	# nn_param_grid = {
	#     'hidden_layer_sizes': [(100,100),(200,200),(200,100),(100, 100, 100), (100, 200, 100), (100, 200, 50)],
	#     'activation': ['tanh'],
	#     'solver': ['sgd'],
	#     'learning_rate_init': [0.001]
	# }

	mlp = MLPClassifier(max_iter=100000, random_state=42)

	nn_grid_search = GridSearchCV(estimator=mlp, param_grid=nn_param_grid, cv=3, verbose=3, n_jobs=-1)

	nn_grid_search.fit(X_train, y_train)

	print("Best Parameters for Neural Network:", nn_grid_search.best_params_)
	print("Best Score for Neural Network:", nn_grid_search.best_score_)

from sklearn.metrics import mean_absolute_error, mean_squared_error
def get_betting_report():
	y_proba = pd.DataFrame(naive_bayes_model(), columns=['prob_0', 'prob_1'])
	# y_proba = pd.DataFrame(xgboost_model(), columns=['prob_0', 'prob_1'])
	x = df_full.iloc[split_index:].reset_index(drop=True)
	df = pd.concat((x, y_proba), axis=1)
	matches = pd.read_csv('csv/odds.csv')
	merged_df = pd.merge(df, matches, on='match_id', how='inner')

	# merged_df['prob_1'] = merged_df['ts_win_prob']
	# merged_df['prob_0'] = 1- merged_df['ts_win_prob']
	merged_df = merged_df[['match_id', 'datetime_x', 'team1_x', 'team2_x', 'win', 'ts_win_prob', 'prob_1', 'prob_0', 't1_odds', 't2_odds', 't1_prob', 't2_prob', 't1_n_prob', 't2_n_prob', 'event_id', 'event_name']]
	merged_df.to_csv('csv/pred_odds.csv')

	correlation = merged_df['prob_1'].corr(merged_df['t1_n_prob'])
	mae = mean_absolute_error(merged_df['prob_1'], merged_df['t1_n_prob'])
	rmse = np.sqrt(mean_squared_error(merged_df['prob_1'], merged_df['t1_n_prob']))

	no_matches = len(merged_df[['match_id']])
	no_bets, wins, losses, profit = 0, 0, 0, 0
	BET_AMOUNT = 1
	for index,row in merged_df.iterrows():
		teams_str = (f"{row['team1_x']} : {row['team2_x']}").ljust(35)
		t1_prediction = row['prob_1']
		t2_prediction = row['prob_0']
		t1_gen_odds = 1/t1_prediction
		t2_gen_odds = 1/t2_prediction
		if t1_gen_odds > 99.99:
			t1_gen_odds = 99.99
		if t2_gen_odds > 99.99:
			t2_gen_odds = 99.99
		t1_b_odds = row['t1_odds']
		t2_b_odds = row['t2_odds']

		print(f"{teams_str} | {f"{t1_b_odds:.2f}".rjust(5)} : {f"{t2_b_odds:.2f}".ljust(5)} | {f"{round(t1_gen_odds, 2):.2f}".rjust(5)} : {f"{round(t2_gen_odds,2):.2f}".ljust(5)}", end = " | ")
		if t1_gen_odds < t1_b_odds:
			no_bets += 1 
			print(f"BETTING {row['team1_x'].ljust(17)}", end = " ")
			if row['win'] == 1:
				wins += 1
				profit += BET_AMOUNT * t1_b_odds
				print(f"WON  + {BET_AMOUNT * t1_b_odds}")
			else:
				losses += 1
				profit -= BET_AMOUNT
				print(f"LOSS - {BET_AMOUNT}")
		elif t2_gen_odds < t2_b_odds:
			print(f"BETTING {row['team2_x'].ljust(17)}", end = " ")
			no_bets += 1
			if row['win'] == 0:
				wins += 1
				profit += BET_AMOUNT * t2_b_odds
				print(f"WON  + {BET_AMOUNT * t2_b_odds}")
			else:
				losses += 1
				profit -= BET_AMOUNT
				print(f"LOSS - {BET_AMOUNT}")
		else:
			print("NO BET                    EVEN 0")
	
	print(len(merged_df))
	
	print(f"& {round(correlation*100,2)} & {round(mae*100,2)} & {round(rmse*100,2)} & {round(100*no_bets/no_matches,2)} & {round(100*wins/no_bets,2)} & {round(100*losses/no_bets,2)} & {round(profit,2)}")

		



if __name__ == "__main__":
	get_betting_report()

	predict = False
	while predict:
		# os.system('cls')
		model_select = input("Select an ML model to evaluate:\n1) Logistic Regression\n2) Random Forest\n3) Support Vector Machine\n4) XGBoost\n5) Gaussian Naive Bayes\n6) MLP Neural Network\n7) k-Nearest Neighbours\n\n> ")
		# os.system('cls')
		match model_select:
			case '1':
				logistic_regression()
			case '2':
				random_forest()
			case '3':
				support_vector_machine()
			case '4':
				xgboost_model()
			case '5':
				naive_bayes_model()
			case '6':
				neural_network_model()
			case '7':
				knn_model()
			case '11':
				logistic_regression_hyperparameter_tuning()
			case '22':
				random_forest_hyperparameter_tuning()
			case '33':
				svm_hyperparameter_tuning()
			case '44':
				xgboost_hyperparameter_tuning()
			case '55':
				naive_bayes_hyperparameter_tuning()
			case '66':
				neural_network_hyperparameter_tuning()
			case '77':
				knn_hyperparameter_tuning()
		pause = input("\nPress ENTER to go back\n\n> ")