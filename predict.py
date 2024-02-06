import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import os
import csv

# df_full = pd.read_csv('csv/df_full.csv', index_col=0)
df_full = pd.read_csv('csv/df_full_diff.csv', index_col=0)
# df_full = pd.read_csv('csv/df_30.csv', index_col=0)
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
X = reduce_features(output=True)

selected_features = ['team1_rank', 'team2_rank', 't1_mu', 't1_sigma', 't2_mu', 't2_sigma', 'ts_win_prob',
       't1_elo', 't2_elo', 'elo_win_prob', 't1_wr', 't2_wr', 'wr_diff', 'map_wr', 'xp_diff',
       'avg_hltv_rating_diff', 'avg_pl_rating_diff', 'avg_pistol_wr_diff']
# X = X[selected_features]

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
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)



    print("Training ACC & Test ACC & Precision & Recall & F1 Score & ROC AUC \\\\ \\hline")
    print(f"& {train_accuracy:.3f} & {test_accuracy:.3f} & {precision:.3f} & {recall:.3f} & {f1:.3f} & {roc_auc:.3f} \\\\")

# Logistic Regression
from sklearn.linear_model import LogisticRegression
def logistic_regression():
	# full: C: 0.0001
	# fs: {'C': 0.001, 'penalty': 'l2', 'solver': 'liblinear'}
	# params = {'C': 0.1, 'penalty': 'l1', 'solver': 'saga'}
	params = {'C': 0.01, 'penalty': 'l2', 'solver': 'liblinear'}
	logistic_regressor = LogisticRegression(**params)
	logistic_regressor.fit(X_train, y_train)
	y_train_pred = logistic_regressor.predict(X_train)
	y_pred = logistic_regressor.predict(X_test)
	y_pred_proba = logistic_regressor.predict_proba(X_test)

	intercept = logistic_regressor.intercept_[0]
	coefficients = logistic_regressor.coef_[0]
	# print(intercept)
	# print(coefficients)


	feature_importances = pd.DataFrame({
		'Feature': X.columns,
		'Importance': coefficients
	}).sort_values(by='Importance', ascending=False)
	print(feature_importances.head(20))
	

	print_stats(y_train, y_train_pred, y_test, y_pred, y_pred_proba[:,-1])

	return y_pred_proba

def logistic_regression_hyperparameter_tuning():
	# Define the parameter grid for Logistic Regression
	lr_param_grid = {
		'C': [0.01, 1, 10],     # inverse of regularization strength
		'penalty': ['l2', 'l1', 'elasticnet'],            # norm used in regularization
		'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']     # alg for optimization
	}
	# lr_param_grid = {
	# 	'C': [0.0075, 0.01, 0.0125],
	# 	'penalty': ['l2'],
	# 	'solver': ['liblinear']
	# }
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

	return y_pred_proba

	# Feature importances
	feature_importances = pd.DataFrame({
		'Feature': X.columns,
		'Importance': rf_classifier.feature_importances_
	}).sort_values(by='Importance', ascending=False)
	print(feature_importances)

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
	params = {'learning_rate': 0.025, 'max_depth': 2, 'n_estimators': 200}
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


	fig, ax = plt.subplots(figsize=(10, 8))
	xgb.plot_importance(xgb_classifier, importance_type='gain', ax=ax, title='Feature Importance', xlabel='F score', max_num_features=18)

	feature_importances = xgb_classifier.get_booster().get_score(importance_type='gain')
	# sorted_features = [feature_names[int(f[1:])] for f in sorted(feature_importances, key=lambda x: feature_importances[x])]
	# ax.set_yticklabels(sorted_features, rotation='horizontal')
	plt.savefig('figures/xgboost_var_imp.png')

	return y_pred_proba

def xgboost_hyperparameter_tuning():
	xgb_param_grid = {
		'n_estimators': [150, 200, 250],    # Number of gradient boosted trees
		'learning_rate': [0.025, 0.05, 0.075],  # Step size shrinkage used in update
		'max_depth': [1, 2, 3],             # Maximum depth of a tree
		# 'subsample': [0.7, 0.8, 0.9],       # Subsample ratio of the training instance
		# 'subsample': [0.7],       # Subsample ratio of the training instance
		# 'colsample_bytree': [0.5] # Subsample ratio of columns when constructing each tree
	}

	xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', enable_categorical=True)

	xgb_grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=xgb_param_grid, cv=4, verbose=2, n_jobs=-1)

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


	# return y_pred_proba
	# proba_df = pd.DataFrame(y_pred_proba, columns=['0_prob', '1_prob'])
	# epsilon = 1e-6
	# proba_df['t1_odds'] = 1 / (proba_df['1_prob'] + epsilon)
	# proba_df['t2_odds'] = 1 / (proba_df['0_prob'] + epsilon)

	# proba_df.index = df_full.iloc[split_index:].index

	# df_info = df_full.iloc[split_index:]
	# result_df = pd.concat([df_info, proba_df], axis=1)
	# result_df.to_csv('csv/predicted_probabilities.csv', index=False)

def neural_network_hyperparameter_tuning():
	nn_param_grid = {
		'hidden_layer_sizes': [(12,12,12), (11,11,11),(10,10,10)],
		# 'hidden_layer_sizes': [(50,),(100,), (200,), (50,50), (100,100)],
		'activation': ['tanh'],
		# 'activation': ['tanh', 'relu'],
		'solver': ['sgd'],
		# 'solver': ['sgd', 'adam'],
		'learning_rate_init': [0.002, 0.001],
		'alpha' : [0.001]
	}

	# nn_param_grid = {
	#     'hidden_layer_sizes': [(100,100),(200,200),(200,100),(100, 100, 100), (100, 200, 100), (100, 200, 50)],
	#     'activation': ['tanh'],
	#     'solver': ['sgd'],
	#     'learning_rate_init': [0.001]
	# }

	mlp = MLPClassifier(max_iter=100000, random_state=42)

	nn_grid_search = GridSearchCV(estimator=mlp, param_grid=nn_param_grid, cv=4, verbose=3, n_jobs=-1)

	nn_grid_search.fit(X_train, y_train)

	print("Best Parameters for Neural Network:", nn_grid_search.best_params_)
	print("Best Score for Neural Network:", nn_grid_search.best_score_)

from sklearn.metrics import mean_absolute_error, mean_squared_error
def get_betting_report():
	# y_proba = pd.DataFrame(knn_model(), columns=['prob_0', 'prob_1'])
	y_proba = pd.DataFrame(knn_model(), columns=['prob_0', 'prob_1'])
	x = df_full.iloc[split_index:].reset_index(drop=True)
	df = pd.concat((x, y_proba), axis=1)
	matches = pd.read_csv('csv/odds.csv')
	merged_df = pd.merge(df, matches, on='match_id', how='inner')
	merged_df = merged_df[['match_id', 'datetime_x', 'team1_x', 'team2_x', 'win', 'prob_1', 'prob_0', 't1_odds', 't2_odds', 't1_prob', 't2_prob', 't1_n_prob', 't2_n_prob', 'event_id', 'event_name']]
	merged_df.to_csv('csv/pred_odds.csv')

	correlation = merged_df['prob_1'].corr(merged_df['t1_n_prob'])
	mae = mean_absolute_error(merged_df['prob_1'], merged_df['t1_n_prob'])
	rmse = np.sqrt(mean_squared_error(merged_df['prob_1'], merged_df['t1_n_prob']))

	# Check which columns have predicted odds better than offered odds
		# check prob_1 > t1_prob, then bet
	# make bets and calculate outcome
	# sum up
	no_bets, wins, losses, profit = 0, 0, 0, 0
	BET_AMOUNT = 1
	for index,row in merged_df.iterrows():
		teams_str = (f"{row['team1_x']} : {row['team2_x']}").ljust(34)
		print(f"{teams_str} | {f"{row['t1_odds']:.2f}".rjust(5)} : {f"{row['t2_odds']:.2f}".rjust(5)} | {round(1/row['prob_1'], 2):.2f} : {round(1/row['prob_0'],2):.2f}", end = " | ")
		if row['prob_1'] > row['t1_prob']:
			no_bets += 1 
			print(f"BETTING {row['team1_x'].ljust(17)}", end = " ")
			if row['win'] == 1:
				wins += 1
				profit += BET_AMOUNT * row['t1_odds']
				print(f"WON  + {BET_AMOUNT * row['t1_odds']}")
			else:
				losses += 1
				profit -= BET_AMOUNT
				print(f"LOSS - {BET_AMOUNT}")
		elif row['prob_0'] > row['t2_prob']:
			print(f"BETTING {row['team2_x'].ljust(17)}", end = " ")
			no_bets += 1
			if row['win'] == 0:
				wins += 1
				profit += BET_AMOUNT * row['t2_odds']
				print(f"WON  + {BET_AMOUNT * row['t2_odds']}")
			else:
				losses += 1
				profit -= BET_AMOUNT
				print(f"LOSS - {BET_AMOUNT}")
		else:
			print("NO BET")
	
	print(len(merged_df))
	
	print(correlation, mae, rmse, end = "\n\n")
	print(no_bets, wins, losses, profit)

		



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