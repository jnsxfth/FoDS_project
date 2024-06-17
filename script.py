import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as sts
from sklearn import tree
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (root_mean_squared_error, mean_absolute_error, r2_score, precision_score, f1_score,
                             roc_auc_score, accuracy_score, recall_score, confusion_matrix, classification_report)
from sklearn.tree import DecisionTreeClassifier
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

# Check if continuous variables are normally distributed
print('Maths:')
for var in num_cols:
    print(f"Shapiro-Wilk for {var}, p-value: {sts.shapiro(data_math[var]).pvalue: .10f}")
print('Portuguese:')
for var in num_cols:
    print(f"Shapiro-Wilk for {var}, p-value: {sts.shapiro(data_port[var]).pvalue: .10f}")

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

# Plot the grade distributions as histograms
for data in [data_math, data_port, data_merged]:
    for column in ['G3', 'G3 passed', '5-level grade']:
        plot_data(data, column)


#Define a linear Regression Algorithm and directly plot the data
def perform_linear_regression_and_plot(data, subject):
    # Perform one-hot encoding on the categorical features
    data_encoded = pd.get_dummies(data, columns=cat_cols, drop_first=True, dtype=int)

    # Construct features (X) and labels (y)
    X = data_encoded.drop(["G3", "G3 passed", "5-level grade", "G1", "G2", 'age'], axis=1)
    y = data_encoded['G3']

    # Update num_cols based on actual numerical columns in X (skip non-existent columns)
    num_cols = X.select_dtypes(include=np.number).columns.tolist()

    # Standardize numerical features using StandardScaler
    sc = StandardScaler()
    X[num_cols] = sc.fit_transform(X[num_cols])

    # Convert pandas DataFrame to numpy array
    X = np.array(X)
    y = np.array(y)

    # Initialize the model using sklearn
    LR = LinearRegression()

    # Perform cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=47)
    r2_scores = cross_val_score(LR, X, y, cv=kf, scoring='r2')
    rmse_scores = -cross_val_score(LR, X, y, cv=kf, scoring='neg_root_mean_squared_error')

    # Train-test split to visualize predictions
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)

    # Make a scatter plot (prediction vs ground truth)
    plt.figure(figsize=(9, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='#008F91')
    plt.xlabel('Actual Grades')
    plt.ylabel('Predicted Grades')
    plt.title(f'Linear Regression: Predicted vs. Actual Grades - {subject}')
    plt.grid(True)

    # Add trendline
    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_test, p(y_test), "r--")

    plt.tight_layout()

    # Save the figure with subject-specific filename
    plt.savefig(os.path.join(f'./output/', f"LinearRegression_{subject}.png"), dpi=100)
    plt.close()

    # Report the performance metrics
    print(f"Performance Metrics for {subject}:")
    print(f"R2 scores: {r2_scores}")
    print(f"Mean R2 score: {np.mean(r2_scores):.3f}")
    print(f"RMSE scores: {rmse_scores}")
    print(f"Mean RMSE: {np.mean(rmse_scores):.3f}")

    # Extract main features (coefficients)
    coefficients = LR.coef_
    feature_importance_df = pd.DataFrame({
        "Feature": data_encoded.drop(["G3", "G3 passed", "5-level grade", "G1", "G2", 'age'], axis=1).columns,
        "Coefficient": coefficients,
        "Absolute Coefficient": np.abs(coefficients)
    }).sort_values(by="Absolute Coefficient", ascending=False)

    print(f"\nTop features for {subject}:")
    print(feature_importance_df.head(7))  # Print top 7 features


# Parameter grid for the hyperparameters
parameter_grid = {
    'n_estimators': [100, 200],
    'max_features': [None, 'sqrt', 'log2'],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 4]
}


# Define Random Forest Regressor Algorithm
def RF_regressor(data, label):
    # Select Features and Labels
    X = data.drop(["G3", "G3 passed", "5-level grade", "G1", "G2", 'age'], axis=1)  # Features
    y = data[label]  # Label
    remaining_cat_cols = [col for col in cat_cols if col in X.columns]
    # one hot encoding
    X = pd.get_dummies(X, columns=remaining_cat_cols, drop_first=True)

    # Metrics for the results
    metrics = {'Root Mean Squared error': [], 'Mean absolute error': [], 'R^2': []}

    # Array for feature importances
    reg_feature_importances = np.zeros(X.shape[1])

    # Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=47)
    fold = 1
    # Conduct Cross Validation
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
        # Conduct GridSearch cross validation
        Reg_GS = GridSearchCV(rf_regressor, parameter_grid, cv=5, verbose=3)
        Reg_GS.fit(X_train, y_train)
        best_params = Reg_GS.best_params_

        # Make predictions
        y_pred = Reg_GS.best_estimator_.predict(X_test)

        # calculate metrics
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # store metrics
        metrics['Root Mean Squared error'].append(rmse)
        metrics['Mean absolute error'].append(mae)
        metrics['R^2'].append(r2)

        # Add feature importances
        reg_feature_importances += Reg_GS.best_estimator_.feature_importances_

        fold += 1
    # Average feature importances across folds
    reg_feature_importances = reg_feature_importances / cv.get_n_splits()

    # Create overview dataframe
    reg_feature_importances_df = pd.DataFrame({'Feature': X.columns, 'Importance': reg_feature_importances})
    reg_feature_importances_df = reg_feature_importances_df.sort_values('Importance', ascending=False)

    # Calculate mean and std of the metrics
    mean_metrics = {key: np.mean(values) for key, values in metrics.items()}
    std_metrics = {key: np.std(values) for key, values in metrics.items()}

    for key, value in mean_metrics.items():
        print('{}: {:.3f} ± {:.3f}'.format(key.capitalize(), value, std_metrics[key]))

    return mean_metrics, std_metrics, best_params, reg_feature_importances_df


def RF_classifier(data, label):
    # Select Features and Labels
    X = data.drop(["G3", "G3 passed", "5-level grade", "G1", "G2", 'age'], axis=1)  # Features
    y = data[label]  # Label
    remaining_cat_cols = [col for col in cat_cols if col in X.columns]
    # One-hot encoding
    X = pd.get_dummies(X, columns=remaining_cat_cols, drop_first=True)

    # Metrics for the results
    metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'ROC-AUC': []}

    # Array for feature importances
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


# Perform linear regression and generate plots for each dataset
perform_linear_regression_and_plot(data_math, "Mathematics")
perform_linear_regression_and_plot(data_port, "Portuguese")
perform_linear_regression_and_plot(data_merged, "Merged")

# Define an overview table for the Random Forest results
overview = pd.DataFrame(
    columns=['Accuracy', 'Accuracy-std', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'ROC-AUC-std', 'MAE', 'MAE-std',
             'RMSE', 'RMSE-std', 'R2',
             'R2-std', 'main feature', 'top 5 features'])
model_parameters = pd.DataFrame(
    columns=['max_depth', 'max_features', 'min_samples_leaf', 'min_samples_split', 'n_estimators'])

label_coding = {'G3 passed': ['G3 passed_Passed', 'G3 passed_Failed'],
                '5-level grade': ['5-level grade_F',
                                  '5-level grade_D', '5-level grade_C', '5-level grade_B', '5-level grade_A']}

features_overview = {}

# conduct several random forest iterations with different labels and different data
for data in [data_math, data_port, data_merged]:
    reg_mean_metrics, reg_std_metrics, reg_best_params, reg_feature_importances_df = RF_regressor(data, 'G3')
    # get the parameters for the different models
    for key in reg_best_params:
        model_parameters.loc[f'{get_dataset_name(data)} Regression - G3', key] = reg_best_params[key]
    # save the performance parameters in the overview table
    overview.loc[f'{get_dataset_name(data)} - G3', 'RMSE'] = reg_mean_metrics['Root Mean Squared error']
    overview.loc[f'{get_dataset_name(data)} - G3', 'RMSE-std'] = reg_std_metrics['Root Mean Squared error']
    overview.loc[f'{get_dataset_name(data)} - G3', 'MAE'] = reg_mean_metrics['Mean absolute error']
    overview.loc[f'{get_dataset_name(data)} - G3', 'MAE-std'] = reg_std_metrics['Mean absolute error']
    overview.loc[f'{get_dataset_name(data)} - G3', 'R2'] = reg_mean_metrics['R^2']
    overview.loc[f'{get_dataset_name(data)} - G3', 'R2-std'] = reg_std_metrics['R^2']
    overview.loc[f'{get_dataset_name(data)} - G3', 'main feature'] = reg_feature_importances_df.Feature.iloc[0]
    overview.loc[f'{get_dataset_name(data)} - G3', 'top 5 features'] = reg_feature_importances_df.Feature.head(
        5).to_list()
    # get the feature importances into a dictionary
    features_overview[f'{get_dataset_name(data)}-G3_features'] = reg_feature_importances_df
    # conduct a classification model for each dataset and label
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

# Plot the R2 values of all RF-Regressions
plt.figure(figsize=(8, 5))
sns.barplot(data=overview.dropna(subset='R2'), x=overview.dropna(subset='R2').index, y='R2',
            yerr=overview.dropna(subset='R2')['R2-std'], color='#008F91')
plt.title(r'$R^2$ values of the three Random Forest Regressors', fontweight='bold', size=14)
plt.ylabel(r'$R^2$ value', fontweight='bold', size=12)
plt.tight_layout()
plt.savefig(f'./output/R^2 values RF_regressor')

# Plot the Accuracy values of all RF-Regressions
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

# Plot a boxplot over the failures and G3 passed in data_port
plt.figure(figsize=(8, 5))
sns.violinplot(x='failures', y='G3 passed', data=data_port, color='#008F91', cut=0, density_norm='area')
plt.title(f'G3 passing over failures in Portuguese data', fontweight='bold', size=14)
plt.xlabel('Failures', fontweight='bold', size=12)
plt.ylabel('G3 passed', fontweight='bold', size=12)
plt.tight_layout()
plt.savefig(f'./output/Portuguese data_failures_over_G3 passed')
plt.close()

# Plot a violinplot over the failures and G3 passed in data_merged
plt.figure(figsize=(8, 5))
sns.violinplot(x='failures', y='G3 passed', data=data_merged, color='#008F91', cut=0, density_norm='area')
plt.title(f'G3 passing over failures in Merged data', fontweight='bold', size=14)
plt.xlabel('Failures', fontweight='bold', size=12)
plt.ylabel('G3 passed', fontweight='bold', size=12)
plt.tight_layout()
plt.savefig(f'./output/Merged data_failures_over_G3 passed')
plt.close()

# Plot a boxplot over the absences and the 5-level grade in data_math
plt.figure(figsize=(8, 5))
sns.boxplot(y='absences', x='5-level grade', data=data_math, color='#008F91', )
plt.title(f'5-level grade over absences in Math data', fontweight='bold', size=14)
plt.ylabel('Absences', fontweight='bold', size=12)
plt.xlabel('5-level grade', fontweight='bold', size=12)
plt.tight_layout()
plt.savefig(f'./output/Math data_failures_over_5-level grade')
plt.close()

### Decision Trees Code ###

# select features and targets
# manually choose which features and targets are generated
features = ['Dalc', 'Walc']
targets = ['G3', 'G3 passed', '5-level grade']
min_samples = 5
x = ['Mathematics', 'Portuguese', 'Merged']
n = 1

### MATHEMATICS ###
for target in targets:

    accuracies = {}

    # separate features and target
    X_math = data_math.loc[:, features]
    y_math = data_math[target]

    # split data into training and test sets
    X_math_train, X_math_test, y_math_train, y_math_test = train_test_split(X_math, y_math, test_size=0.2,
                                                                            random_state=42)

    # train the decision tree classifier
    clf_math = DecisionTreeClassifier(min_samples_leaf=min_samples, random_state=42)
    clf_math.fit(X_math_train, y_math_train)

    # make predictions
    y_math_pred = clf_math.predict(X_math_test)

    # evaluate the model
    # accuracy
    accuracy_math = accuracy_score(y_math_test, y_math_pred)
    print(f"Accuracy (Mathematics) - {features[0]}-{target}: {accuracy_math}")
    accuracies['Math'] = accuracy_math

    # confusion matrix
    conf_matrix_math = confusion_matrix(y_math_test, y_math_pred)
    print(f"Confusion Matrix (Mathematics) - {target}:")
    print(conf_matrix_math)

    # make heatmap

    # classification report
    class_report_math = classification_report(y_math_test, y_math_pred, output_dict=True)
    plt.figure(n, figsize=(8, 6))
    n += 1
    df = pd.DataFrame(class_report_math).transpose()
    df = df.loc[:, ['precision', 'recall']]
    sns.heatmap(df.iloc[:-1, :])
    if len(features) == 1:
        plt.title(f'Heatmap Classification Report Math {features[0]} predicting {target}')
    else:
        plt.title(f'Heatmap Classification Report Math {features[0]} and {features[1]} predicting {target}')
    plt.savefig(f'./output/class reports/Classification Report Math {features[0]} - {target}')

    # visualize
    plt.figure(n, figsize=(30, 20))
    n += 1
    tree.plot_tree(clf_math, filled=True, feature_names=X_math.columns, class_names=clf_math.classes_.astype(str))
    plt.title(f'Decision Tree Mathematics - {target}')
    plt.savefig(f'/output/Decision Trees/decision tree math - {features[0]}-{target}.jpg', format='jpg')

    ### PORTUGUESE ###

    # separate features and target
    X_port = data_port.loc[:, features]
    y_port = data_port[target]

    # split data into training and test set
    X_port_train, X_port_test, y_port_train, y_port_test = train_test_split(X_port, y_port, test_size=0.2,
                                                                            random_state=42)

    # train the decision tree classifier
    clf_port = DecisionTreeClassifier(min_samples_leaf=min_samples, random_state=42)
    clf_port.fit(X_port_train, y_port_train)

    # make predictions
    y_port_pred = clf_port.predict(X_port_test)

    # evaluate the model
    # accuracy
    accuracy_port = accuracy_score(y_port_test, y_port_pred)
    print(f"Accuracy (Portuguese) - {features[0]}-{target}: {accuracy_port}")
    accuracies['Port'] = accuracy_port

    # confusion matrix
    conf_matrix_port = confusion_matrix(y_port_test, y_port_pred)
    print(f"Confusion Matrix (Portuguese) - {target}:")
    print(conf_matrix_port)

    # classification report
    class_report_port = classification_report(y_port_test, y_port_pred, output_dict=True)
    plt.figure(n, figsize=(8, 6))
    n += 1
    df = pd.DataFrame(class_report_port).transpose()
    df = df.loc[:, ['precision', 'recall']]
    sns.heatmap(df.iloc[:-1, :])
    if len(features) == 1:
        plt.title(f'Heatmap Classification Report Port {features[0]} predicting {target}')
    else:
        plt.title(f'Heatmap Classificatino Report Port {features[0]} and {features[1]} predicting {target}')
    plt.savefig(f'./output/class reports/Classification Report Port {features} - {target}')

    # visualize
    plt.figure(n, figsize=(30, 20))
    n += 1
    tree.plot_tree(clf_port, filled=True, feature_names=X_port.columns, class_names=clf_port.classes_.astype(str))
    plt.title(f'Decision Tree Portuguese - {target}')
    plt.savefig(f'./output/Decision Trees/decision tree port - {features[0]}-{target}.png', format='png')

    ### MIXED

    # seperate features and target(s)
    X_merged = data_merged.loc[:, features]
    y_merged = data_merged[target]

    # split data into training and test set
    X_merged_train, X_merged_test, y_merged_train, y_merged_test = train_test_split(X_merged, y_merged, test_size=0.2,
                                                                                    random_state=42)

    # train the decision tree classifier
    clf_merged = DecisionTreeClassifier(min_samples_leaf=min_samples, random_state=42)
    clf_merged.fit(X_merged_train, y_merged_train)

    # make predictions
    y_merged_pred = clf_merged.predict(X_merged_test)

    # evaluate the model
    # accuracy
    accuracy_merged = accuracy_score(y_merged_test, y_merged_pred)
    print(f"Accuracy (Merged) - {features[0]}-{target}: {accuracy_merged}")
    accuracies['Merged'] = accuracy_merged

    # confusion matrix
    conf_matrix_merged = confusion_matrix(y_merged_test, y_merged_pred)
    print(f"Confusion Matrix (Merged) - {target}:")
    print(conf_matrix_merged)

    # classification report
    class_report_merged = classification_report(y_merged_test, y_merged_pred, output_dict=True)
    plt.figure(n, figsize=(8, 6))
    n += 1
    df = pd.DataFrame(class_report_merged).transpose()
    df = df.loc[:, ['precision', 'recall']]
    sns.heatmap(df.iloc[:-1, :])
    if len(features) == 1:
        plt.title(f'Heatmap Classification Report Merged {features[0]} predicting {target}')
    else:
        plt.title(f'Heatmap Classification Report Merged {features[0]} and {features[1]} predicting {target}')
    plt.savefig(f'./output/class reports/Classification Report Merged {features} - {target}')

    # visualize
    plt.figure(n, figsize=(50, 35))
    n += 1
    tree.plot_tree(clf_merged, filled=True, feature_names=X_merged.columns, class_names=clf_merged.classes_.astype(str))
    plt.title(f'Decision Tree Merged - {features[0]} - {target}')
    plt.savefig(f'./output/Decision Trees/decision tree merged - {features[0]}-{target}.png', format='png')

    plt.figure(n, figsize=(8, 6))
    n += 1
    y = []
    for i in accuracies:
        y.append(accuracies[i])
    plt.bar(x, y, color='#008F91')
    plt.ylim(min(y) - 0.15, max(y) + 0.15)
    if len(features) == 2:
        plt.title(f'Accuracies of {features[0]} and {features[1]} predicting {target}')
    else:
        plt.title(f'Accuracies of {features[0]} predicting {target}')
    plt.savefig(f'./output/plots/accuracies {features} predicting {target}')

"""Finale Version 17.06. 5x5"""
