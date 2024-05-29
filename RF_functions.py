from script import *
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (root_mean_squared_error, mean_absolute_error, r2_score,
                             precision_score, f1_score, roc_auc_score, accuracy_score, recall_score)
import numpy as np


#Define Random Forest Regressor Algorithm
def RF_regressor(data, label):
    # parameter grid for the hyperparameters
    parameter_grid = {'n_estimators': [100, 200],
                      'max_features': [None, 'sqrt', 'log2'],
                      'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4], 'bootstrap': [True, False]}

    # Select Features and Labels
    X = data.drop(["G3", "G3 passed", "5-level grade", "G1", "G2", 'age'], axis=1)  # Features
    y = data[label]  # Label
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)  #one hot encoding

    # Metrics for the results
    metrics = {'Root Mean Squared error': [], 'Mean absolute error': [], 'R^2': []}

    reg_feature_importances = np.zeros(X.shape[1])

    # Cross Validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=47)
    fold = 1
    #Conduct Cross Validation
    for train_index, test_index in cv.split(X, y):
        print('Fold: ', fold)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # create Random forest regressor
        rf_regressor = RandomForestRegressor(random_state=47)
        #Conduct GridSearch cross validation
        Reg_GS = GridSearchCV(rf_regressor, parameter_grid, cv=3, verbose=3)
        Reg_GS.fit(X_train, y_train)
        best_params = Reg_GS.best_params_

        #Make predictions
        y_pred = Reg_GS.best_estimator_.predict(X_test)

        #calculate metrics
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        #store metrics
        metrics['Root Mean Squared error'].append(rmse)
        metrics['Mean absolute error'].append(mae)
        metrics['R^2'].append(r2)

        #Add feature importances
        reg_feature_importances += Reg_GS.best_estimator_.feature_importances_

        fold += 1
    #Average feature importances across folds
    reg_feature_importances = reg_feature_importances / cv.get_n_splits()

    #Create overview dataframe
    reg_feature_importances_df = pd.DataFrame({'Feature': X.columns, 'Importance': reg_feature_importances})
    reg_feature_importances_df = reg_feature_importances_df.sort_values('Importance', ascending=False)

    #Calculate mean and std of the metrics
    mean_metrics = {key: np.mean(values) for key, values in metrics.items()}
    std_metrics = {key: np.std(values) for key, values in metrics.items()}

    for key, value in mean_metrics.items():
        print('{}: {:.3f} ± {:.3f}'.format(key.capitalize(), value, std_metrics[key]))

    return mean_metrics, std_metrics, best_params, reg_feature_importances_df

def RF_classifier(data, label):
    # Parameter grid for the hyperparameters
    parameter_grid = {
        'n_estimators': [100, 200],
        'max_features': [None, 'sqrt', 'log2'],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Select Features and Labels
    X = data.drop(["G3", "G3 passed", "5-level grade", "G1", "G2", 'age'], axis=1)  # Features
    y = data[label]  # Label
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)  # One-hot encoding

    # Metrics for the results
    metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ROC-AUC': []}

    clf_feature_importances = np.zeros(X.shape[1])

    # Cross Validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=47)
    fold = 1

    # Conduct Cross Validation
    for train_index, test_index in cv.split(X, y):
        print('Fold:', fold)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create Random Forest classifier
        rf_classifier = RandomForestClassifier(random_state=47)

        # Conduct GridSearch cross validation

        clf_GS = GridSearchCV(rf_classifier, parameter_grid, cv=3, verbose=3)
        clf_GS.fit(X_train, y_train)
        best_params = clf_GS.best_params_

        # Make predictions
        y_pred = clf_GS.best_estimator_.predict(X_test)
        y_prob = clf_GS.best_estimator_.predict_proba(X_test)  # Probabilities for ROC-AUC

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        if len(np.unique(y))>2:
            roc_auc = roc_auc_score(y_test, y_prob, average='weighted', multi_class='ovr')
        else:
            roc_auc = roc_auc_score(y_test, y_prob[:,1], average='weighted', multi_class='ovr')

        # Store metrics
        metrics['Accuracy'].append(accuracy)
        metrics['Precision'].append(precision)
        metrics['Recall'].append(recall)
        metrics['F1 Score'].append(f1)
        metrics['ROC-AUC'].append(roc_auc)

        # Add feature importances
        clf_feature_importances += clf_GS.best_estimator_.feature_importances_

        fold += 1

    # Average feature importances across folds
    clf_feature_importances /= cv.get_n_splits()

    # Create overview dataframe
    clf_feature_importances_df = pd.DataFrame({'Feature': X.columns, 'Importance': clf_feature_importances})
    clf_feature_importances_df = clf_feature_importances_df.sort_values('Importance', ascending=False)

    # Calculate mean and std of the metrics
    mean_metrics = {key: np.mean(values) for key, values in metrics.items()}
    std_metrics = {key: np.std(values) for key, values in metrics.items()}

    for key, value in mean_metrics.items():
        print('{}: {:.3f} ± {:.3f}'.format(key, value, std_metrics[key]))

    return mean_metrics, std_metrics, best_params, clf_feature_importances_df

