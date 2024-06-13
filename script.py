import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (root_mean_squared_error, mean_absolute_error, r2_score,
                             precision_score, f1_score, roc_auc_score, accuracy_score, recall_score)
from imblearn.over_sampling import SMOTE

# import data
data_math = pd.read_csv('./data/Maths.csv')
data_port = pd.read_csv('./data/Portuguese.csv')

# Check for missing values
missing_val_math = data_math.isnull().sum().sum()
missing_val_port = data_port.isnull().sum().sum()

# Check for duplicated values
dupl_port = data_port[data_port.duplicated()].size
dupl_math = data_math[data_math.duplicated()].size

# Create a new column that indicates if G3 is passed or not
data_math['G3 passed'] = data_math['G3'].apply(lambda x: 'Passed' if x >= 10 else 'Failed')
data_port['G3 passed'] = data_port['G3'].apply(lambda x: 'Passed' if x >= 10 else 'Failed')

# Create a new column that converts the grades into a 5-level scale
bins = [0, 10, 12, 14, 16, 21]
labels = ['F', 'D', 'C', 'B', 'A']
data_math['5-level grade'] = pd.cut(data_math['G3'], bins=bins, labels=labels, right=False)
data_port['5-level grade'] = pd.cut(data_port['G3'], bins=bins, labels=labels, right=False)

# Get a description of the numerical values of the datasets
data_math_description = data_math.describe()
data_port_description = data_port.describe()

# Define categorical columns
cat_cols = ['sex', 'school', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
            'traveltime', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'schoolsup', 'famsup',
            'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'studytime', 'G3 passed',
            '5-level grade']
# Define numerical columns
num_cols = ['age', 'absences', 'failures', 'G1', 'G2', 'G3']

# Set categorical columns to datatype category
data_math[cat_cols] = data_math[cat_cols].astype('category')
data_port[cat_cols] = data_port[cat_cols].astype('category')

"""#Check if continuous variables are normally distributed
print('Maths:')
for var in num_cols:
    print(f"Shapiro-Wilk for {var}, p-value: {sts.shapiro(data_math[var]).pvalue: .10f}")
print('Portuguese:')
for var in num_cols:
    print(f"Shapiro-Wilk for {var}, p-value: {sts.shapiro(data_port[var]).pvalue: .10f}")"""

# If Shapiro significant => data not normally distributed -> they are all not normally distributed

# Combine the datasets
data_merged_raw = pd.concat([data_math, data_port])
data_merged = data_merged_raw.drop_duplicates(
    subset=data_math.columns.drop(['G1', 'G2', 'G3', 'G3 passed', '5-level grade']).tolist())

# create a dictionary for the data-frame names
data_titles = {'Math data': data_math, 'Portuguese data': data_port, 'Merged data': data_merged}


# create a function that allows to get the name of a dataframe
def get_dataset_name(df):
    for name, dataset in data_titles.items():
        if df.equals(dataset):
            return name
    return None


# Function to plot a variable as histogram

def plot_data(data, column):
    plt.figure(figsize=(10, 8))
    sns.histplot(data[column], binwidth=1, discrete=True, color='#008F91')
    plt.title(f"Distribution of\n{titles[column]} in {get_dataset_name(data)}", size=14, fontweight='bold')
    plt.xlabel(xlabels[column], fontweight='bold', size=16)
    plt.ylabel('Count', fontweight='bold', size=16)
    plt.savefig(f'./output/{get_dataset_name(data)}/{column}_histplot')
    plt.close()


# Function to plot the variable as boxplot
def boxplot_data(data, column):
    plt.figure(figsize=(10, 8))
    sns.boxplot(x=data[column], color='#008F91')
    plt.title(f'Distribution of\n {titles[column]} in {get_dataset_name(data)}', size=14, fontweight='bold')
    plt.xlabel(xlabels[column], fontweight='bold', size=16)
    plt.ylabel('Count', fontweight='bold', size=16)
    plt.savefig(f'./output/{get_dataset_name(data)}/{column}_boxplot')
    plt.close()


# Plot the distributions of the columns as histograms to get an overview
titles = {
    "school": "visited school",
    "sex": "sex",
    "address": "home environment",
    "age": "age",
    "famsize": "Family size",
    "Pstatus": "Parental status",
    "Medu": "mother's education",
    "Fedu": "father's education",
    "Mjob": "mother's job",
    "Fjob": "father's job",
    "reason": "reason to choose school",
    "guardian": "student's guardian",
    "traveltime": "student's travel-time to school",
    "studytime": "student's weekly studytime",
    "failures": "student's past class failures",
    "schoolsup": "extra educational support",
    "famsup": "family educational support",
    "paid": "extra paid classes within the course subject",
    "activities": "extra-curricular activities",
    "nursery": "attended nursery school",
    "higher": "wants to take higher education",
    "internet": "internet access at home",
    "romantic": "romantic relationship",
    "famrel": "quality of family relationships",
    "freetime": "free time after school",
    "goout": "frequency of going out with friends",
    "Dalc": "workday alcohol consumption",
    "Walc": "weekend alcohol consumption",
    "health": "current health status",
    "absences": "number of school absences",
    "G1": "first grade",
    "G2": "second grade",
    "G3": "final grade",
    "G3 passed": "G3 passed",
    "5-level grade": "5-level grade"
}
xlabels = {
    "school": "visited school",
    "sex": "sex",
    "address": "home environment [urban or rural]",
    "age": "Age [years]",
    "famsize": "Family size [<=3 or >3]",
    "Pstatus": "Parental status [together or apart]",
    "Medu": "mother's education [",
    "Fedu": "father's education",
    "Mjob": "mother's job",
    "Fjob": "father's job",
    "reason": "reason to choose school",
    "guardian": "student's guardian",
    "traveltime": "student's travel-time to school",
    "studytime": "student's weekly studytime",
    "failures": "student's past class failures",
    "schoolsup": "extra educational support",
    "famsup": "family educational support",
    "paid": "extra paid classes within the course subject",
    "activities": "extra-curricular activities",
    "nursery": "attended nursery school",
    "higher": "wants to take higher education",
    "internet": "internet access at home",
    "romantic": "romantic relationship",
    "famrel": "quality of family relationships [very bad to excellent]",
    "freetime": "free time after school [very low to very high]",
    "goout": "frequency of going out with friends [very low to very high]",
    "Dalc": "workday alcohol consumption [very low to very high]",
    "Walc": "weekend alcohol consumption [very low to very high]",
    "health": "current health status [very bad to very good]",
    "absences": "number of school absences",
    "G1": "first grade",
    "G2": "second grade",
    "G3": "final grade",
    "G3 passed": "whether G3 was passed",
    "5-level grade": "G3 converted to 5-level grade"
}

# Create paths directories for plots
for path in [f'{os.getcwd()}/output',
             f'{os.getcwd()}/output/{get_dataset_name(data_math)}',
             f'{os.getcwd()}/output/{get_dataset_name(data_port)}',
             f'{os.getcwd()}/output/{get_dataset_name(data_merged)}']:
    if not os.path.exists(path):
        os.makedirs(path)

# Plot some distributions as histograms
for data in [data_math, data_port, data_merged]:
    for column in ['G3', 'G3 passed', '5-level grade']:
        plot_data(data, column)

""""# Plot numerical distributions as boxplots
for data in [data_math, data_port]:
    for column in ['age']:
        boxplot_data(data, column)"""


# Define Random Forest Regressor Algorithm
def RF_regressor(data, label):
    # parameter grid for the hyperparameters
    parameter_grid = {'n_estimators': [100, 200],
                      'max_features': [None, 'sqrt', 'log2'],
                      'max_depth': [10, 20], 'min_samples_split': [2, 5],
                      'min_samples_leaf': [1, 2, 4]}

    # Select Features and Labels
    X = data.drop(["G3", "G3 passed", "5-level grade", "G1", "G2", 'age'], axis=1)  # Features
    y = data[label]  # Label
    remaining_cat_cols = [col for col in cat_cols if col in X.columns]
    X = pd.get_dummies(X, columns=remaining_cat_cols, drop_first=True)  #one hot encoding

    # Metrics for the results
    metrics = {'Root Mean Squared error': [], 'Mean absolute error': [], 'R^2': []}

    reg_feature_importances = np.zeros(X.shape[1])

    # Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=47)
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
        Reg_GS = GridSearchCV(rf_regressor, parameter_grid, cv=5, verbose=3)
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
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 4]
    }

    # Select Features and Labels
    X = data.drop(["G3", "G3 passed", "5-level grade", "G1", "G2", 'age'], axis=1)  # Features
    y = data[label]  # Label
    remaining_cat_cols = [col for col in cat_cols if col in X.columns]
    X = pd.get_dummies(X, columns=remaining_cat_cols, drop_first=True)  # One-hot encoding

    # Metrics for the results
    metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ROC-AUC': []}

    clf_feature_importances = np.zeros(X.shape[1])

    # Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=47)
    fold = 1

    # Conduct Cross Validation
    for train_index, test_index in cv.split(X, y):
        print('Fold:', fold)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Apply SMOTE to balance the data
        smote = SMOTE(random_state=47)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create Random Forest classifier
        rf_classifier = RandomForestClassifier(random_state=47)

        # Conduct GridSearch cross validation
        clf_GS = GridSearchCV(rf_classifier, parameter_grid, cv=5, verbose=3)
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
        if len(np.unique(y)) > 2:
            roc_auc = roc_auc_score(y_test, y_prob, average='weighted', multi_class='ovr')
        else:
            roc_auc = roc_auc_score(y_test, y_prob[:, 1], average='weighted', multi_class='ovr')

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


#Define an overview table for the model results
overview = pd.DataFrame(
    columns=['Accuracy', 'Accuracy-std', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'ROC-AUC-std', 'MAE', 'MAE-std', 'RMSE', 'RMSE-std', 'R2',
             'R2-std', 'main feature', 'top 5 features'])
model_parameters = pd.DataFrame(
    columns=['max_depth', 'max_features', 'min_samples_leaf', 'min_samples_split', 'n_estimators'])

label_coding = {'G3 passed': ['G3 passed_Passed', 'G3 passed_Failed'],
                '5-level grade': ['5-level grade_F',
                                  '5-level grade_D', '5-level grade_C', '5-level grade_B', '5-level grade_A']}

features_overview = {}

#conduct several random forest iterations with different labels and different data
for data in [data_math, data_port, data_merged]:
    reg_mean_metrics, reg_std_metrics, reg_best_params, reg_feature_importances_df = RF_regressor(data, 'G3')
    #get the parameters for the different models
    for key in reg_best_params:
        model_parameters.loc[f'{get_dataset_name(data)} Regression - G3', key] = reg_best_params[key]
    #save the performance parameters in the overview table
    overview.loc[f'{get_dataset_name(data)} - G3', 'RMSE'] = reg_mean_metrics['Root Mean Squared error']
    overview.loc[f'{get_dataset_name(data)} - G3', 'RMSE-std'] = reg_std_metrics['Root Mean Squared error']
    overview.loc[f'{get_dataset_name(data)} - G3', 'MAE'] = reg_mean_metrics['Mean absolute error']
    overview.loc[f'{get_dataset_name(data)} - G3', 'MAE-std'] = reg_std_metrics['Mean absolute error']
    overview.loc[f'{get_dataset_name(data)} - G3', 'R2'] = reg_mean_metrics['R^2']
    overview.loc[f'{get_dataset_name(data)} - G3', 'R2-std'] = reg_std_metrics['R^2']
    overview.loc[f'{get_dataset_name(data)} - G3', 'main feature'] = reg_feature_importances_df.Feature.iloc[0]
    overview.loc[f'{get_dataset_name(data)} - G3', 'top 5 features'] = reg_feature_importances_df.Feature.head(
        5).to_list()
    #get the feature importances into a dictionary
    features_overview[f'{get_dataset_name(data)}-G3_features'] = reg_feature_importances_df
    #conduct a classification model for each dataset and label
    for label in ['G3 passed', '5-level grade']:
        clf_mean_metrics, clf_std_metrics, clf_best_params, clf_feature_importances_df = RF_classifier(data, label)
        for key in clf_best_params:
            model_parameters.loc[f'{get_dataset_name(data)} Classification - {label}', key] = clf_best_params[key]
        # save the performance parameters in the overview table
        overview.loc[f'{get_dataset_name(data)} - {label}', 'Accuracy'] = clf_mean_metrics['Accuracy']
        overview.loc[f'{get_dataset_name(data)} - {label}', 'Accuracy-std'] = clf_std_metrics['Accuracy']
        overview.loc[f'{get_dataset_name(data)} - {label}', 'Precision'] = clf_mean_metrics['Precision']
        overview.loc[f'{get_dataset_name(data)} - {label}', 'F1'] = clf_mean_metrics['F1 Score']
        overview.loc[f'{get_dataset_name(data)} - {label}', 'Recall'] = clf_mean_metrics['Recall']
        overview.loc[f'{get_dataset_name(data)} - {label}', 'ROC-AUC'] = clf_mean_metrics['ROC-AUC']
        overview.loc[f'{get_dataset_name(data)} - {label}', 'ROC-AUC-std'] = clf_std_metrics['ROC-AUC']
        overview.loc[f'{get_dataset_name(data)} - {label}', 'main feature'] = clf_feature_importances_df.Feature.iloc[0]
        overview.loc[f'{get_dataset_name(data)} - {label}', 'top 5 features'] = clf_feature_importances_df.Feature.head(
            5).to_list()
        # get the feature importances into a dictionary
        features_overview[f'{get_dataset_name(data)}-{label}_features'] = clf_feature_importances_df

#Plot the R2 values of all RF-Regressions
plt.figure(figsize=(8, 5))
sns.barplot(data=overview.dropna(subset='R2'), x=overview.dropna(subset='R2').index, y='R2',
            yerr=overview.dropna(subset='R2')['R2-std'], color='#008F91')
plt.title(r'$R^2$ values of the three Random Forest Regressors', fontweight='bold', size=14)
plt.ylabel(r'$R^2$ value', fontweight='bold', size=12)
plt.tight_layout()
plt.savefig(f'./output/R^2 values RF_regressor')

#Plot the Accuracy values of all RF-Regressions
plt.figure(figsize=(8, 5))
sns.barplot(data=overview.dropna(subset='Accuracy'), x=overview.dropna(subset='Accuracy').index, y='Accuracy',
            yerr=overview.dropna(subset='Accuracy')['Accuracy-std'], color='#008F91')
plt.title('Accuracy values of the three Random Forest Classifiers', fontweight='bold', size=14)
plt.ylabel('Accuracy value', fontweight='bold', size=12)
plt.xticks(rotation=66)
plt.tight_layout()
plt.savefig(f'./output/Accuracy values RF_classifier')

# Plot a boxplot with regression line over the failures and G3 in data_port
plt.figure(figsize=(8, 5))
sns.boxplot(x='failures', y='G3', data=data_port, color='#008F91', )
sns.regplot(x='failures', y='G3', data=data_port, scatter=False, ci=None, color='red',
            line_kws={'label': 'Regression Line'})
plt.title(f'G3 score over failures in Portuguese data with regression line', fontweight='bold', size=14)
plt.xlabel('Failures', fontweight='bold', size=12)
plt.ylabel('G3 score', fontweight='bold', size=12)
plt.tight_layout()
plt.savefig(f'./output/Portuguese data_failures_over_G3')
plt.close()

# Plot a boxplot with regression line over the failures and G3 in data_merged
plt.figure(figsize=(8, 5))
sns.boxplot(x='failures', y='G3', data=data_merged, color='#008F91', )
sns.regplot(x='failures', y='G3', data=data_merged, scatter=False, ci=None, color='red',
            line_kws={'label': 'Regression Line'})
plt.title(f'G3 score over failures in Merged data with regression line', fontweight='bold', size=14)
plt.xlabel('Failures', fontweight='bold', size=12)
plt.ylabel('G3 score', fontweight='bold', size=12)
plt.tight_layout()
plt.savefig(f'./output/Merged data_failures_over_G3')
plt.close()

#Plot a boxplot over the failures and G3 passed in data_port
plt.figure(figsize=(8, 5))
sns.violinplot(x='failures', y='G3 passed', data=data_port, color='#008F91', cut=0, density_norm='area')
plt.title(f'G3 passing over failures in Portuguese data', fontweight='bold', size=14)
plt.xlabel('Failures', fontweight='bold', size=12)
plt.ylabel('G3 passed', fontweight='bold', size=12)
plt.tight_layout()
plt.savefig(f'./output/Portuguese data_failures_over_G3 passed')
plt.close()

#Plot a violinplot over the failures and G3 passed in data_merged
plt.figure(figsize=(8, 5))
sns.violinplot(x='failures', y='G3 passed', data=data_merged, color='#008F91', cut=0, density_norm='area')
plt.title(f'G3 passing over failures in Merged data', fontweight='bold', size=14)
plt.xlabel('Failures', fontweight='bold', size=12)
plt.ylabel('G3 passed', fontweight='bold', size=12)
plt.tight_layout()
plt.savefig(f'./output/Merged data_failures_over_G3 passed')
plt.close()

#Plot a boxplot over the absences and the 5-level grade in data_math
plt.figure(figsize=(8, 5))
sns.boxplot(y='absences', x='5-level grade', data=data_math, color='#008F91', )
plt.title(f'5-level grade over absences in Math data', fontweight='bold', size=14)
plt.ylabel('Absences', fontweight='bold', size=12)
plt.xlabel('5-level grade', fontweight='bold', size=12)
plt.tight_layout()
plt.savefig(f'./output/Math data_failures_over_5-level grade')
plt.close()

"""Finale Version 11.06. 5x5"""
