import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as sts
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (root_mean_squared_error, mean_absolute_error, r2_score,
                             precision_score, f1_score, roc_auc_score, roc_curve)

#import data
data_math = pd.read_csv('./data/Maths.csv')
data_port = pd.read_csv('./data/Portuguese.csv')

#Check for missing values
missing_val_math = data_math.isnull().sum().sum()
missing_val_port = data_port.isnull().sum().sum()

#Check for duplicated values
dupl_port = data_port[data_port.duplicated()].size
dupl_math = data_math[data_math.duplicated()].size

#Get a description of the numerical values of the datasets
data_math_description = data_math.describe()
data_port_description = data_port.describe()
"""*Explaining Columns*

school = (GP or MS)
sex = (F or M)
age = (15-22)
address = urban or rural (U or R)
famsize = 'less or equal to 3' or 'greater than 3' (LE3 or GT3)
Psatus = parent's cohabitation status: together or apart (T or A)
Medu = mother's education: none, primary, 5th to 9th grade, secondary, higher (0-4)
Fedu = father's education: none, primary, 5th to 9th grade, secondary, higher (0-4)
Mjob = mother's job: (teacher, health, services, at_home, other)
Fjob = father's job: (teacher, health, services, at_home, other)
reason = reason to choose school: close to home, reputation, course preverence, other (home, reputation, course, other)
guardian = student's guardian (mother, father, other)
traveltime = home to school travel time (1=1-<15min, 2=15-30min, 3=30-60min, 4=>1h)
studytime = weekly study time (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)
failures = number of past class failures (n if 1<=n<3, else 4)
schoolsup = extra educational support (yes or no)
famsup = family educational support (yes or no)
paid = extra paid classes within the course subject (yes or no)
activities = extra-curricular activities (yes or no)
nursery = attended nursery school (yes or no)
higher = wants to take higher education (yes or no)
internet = internet access at home (yes or no)
romantic = with a romantic relationship (yes or no)
famrel = quality of family relationships (1=very bad to 5=excellent)
freetime = free time after school (1= very low to 5=very high)
goout = going out with friends (1=very low to 5=very high)
Dalc = workday alcohol consumption (1=very low to 5=very high)
Walc = weekend alcohol consumption (1=very low to 5=very high)
health = current health status (1=very bad to 5=very good)
absences = number of school absences (0-93)
G1-G3 = first/second/final grade (0-20)
"""

#Define categorical columns
cat_cols = ['sex', 'school', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
            'traveltime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
            'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'studytime',]
#Define numerical columns
num_cols = ['age','absences', 'G1', 'G2', 'G3']

#Set categorical columns to datatype category
data_math[cat_cols] = data_math[cat_cols].astype('category')
data_port[cat_cols] = data_port[cat_cols].astype('category')

#rename the education categories
def rename_education(data, parent):
    data[parent] = data[parent].cat.rename_categories(
        {0: "none", 1: "primary education", 2: "5th to 9th", 3: "secondary education", 4: "higher education"})
rename_education(data_math, 'Fedu')
rename_education(data_port, 'Fedu')
rename_education(data_math, 'Medu')
rename_education(data_port, 'Medu')

#rename the studytime categories
data_math["studytime"] = data_math["studytime"].cat.rename_categories(
        {1: "<=2h", 2: "2-5h", 3: "5-10h", 4: ">=10h"})
data_port["studytime"] = data_port["studytime"].cat.rename_categories(
        {1: "<=2h", 2: "2-5h", 3: "5-10h", 4: ">=10h"})
#rename the traveltime categories
data_math["traveltime"] = data_math["traveltime"].cat.rename_categories(
        {1: "1-<15min", 2: "15-30min", 3: "30-60min", 4: ">1h"})
data_port["traveltime"] = data_port["traveltime"].cat.rename_categories(
        {1: "1-<15min", 2: "15-30min", 3: "30-60min", 4: ">1h"})

#Plot the distributions of the columns as histograms to get an overview
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
    "nursery" : "attended nursery school",
    "higher" : "wants to take higher education",
    "internet": "internet access at home",
    "romantic" : "romantic relationship",
    "famrel" : "quality of family relationships",
    "freetime" : "free time after school",
    "goout" : "frequency of going out with friends",
    "Dalc" : "workday alcohol consumption",
    "Walc" : "weekend alcohol consumption",
    "health" : "current health status",
    "absences": "number of school absences",
    "G1": "first grade",
    "G2": "second grade",
    "G3": "final grade"
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
    "nursery" : "attended nursery school",
    "higher" : "wants to take higher education",
    "internet": "internet access at home",
    "romantic" : "romantic relationship",
    "famrel" : "quality of family relationships [very bad to excellent]",
    "freetime" : "free time after school [very low to very high]",
    "goout" : "frequency of going out with friends [very low to very high]",
    "Dalc" : "workday alcohol consumption [very low to very high]",
    "Walc" : "weekend alcohol consumption [very low to very high]",
    "health" : "current health status [very bad to very good]",
    "absences": "number of school absences",
    "G1": "first grade",
    "G2": "second grade",
    "G3": "final grade"
}
def plot_data_math(column):
    plt.figure(figsize=(10,8))
    sns.histplot(data_math[column], binwidth=1)
    plt.title(f"Distribution of\n{titles[column]} in Maths data")
    plt.xlabel(xlabels[column])
    plt.savefig(f'./output/math/{column}_distribution_math')

def plot_data_port(column):
    plt.figure(figsize=(10,8))
    sns.histplot(data_port[column], binwidth=1)
    plt.title(f"Distribution of\n{titles[column]} in Portuguese data")
    plt.xlabel(xlabels[column])
    plt.savefig(f'./output/port/{column}_distribution_port')

def boxplot_data_port(column):
    plt.figure(figsize=(10,8))
    sns.boxplot(x = data_port[column])
    plt.title(f'Distribution of\n {titles[column]} in Portuguese data')
    plt.xlabel(xlabels[column])
    plt.savefig(f'./output/port/{column}_boxplot_port')

def boxplot_data_math(column):
    plt.figure(figsize=(10,8))
    sns.boxplot(x = data_math[column])
    plt.title(f'Distribution of\n {titles[column]} in Maths data')
    plt.xlabel(xlabels[column])
    plt.savefig(f'./output/math/{column}_boxplot_math')


#Plot all distributions as histograms
for i in data_math.columns:
    plot_data_math(i)
    plot_data_port(i)
#Plot numerical distributions as boxplots
for i in num_cols:
    boxplot_data_port(i)
    boxplot_data_math(i)

#Check if continuous variables are normally distributed
print('Maths:')
for var in num_cols:
    print(f"Shapiro-Wilk for {var}, p-value: {sts.shapiro(data_math[var]).pvalue: .10f}")
print('Portuguese:')
for var in num_cols:
    print(f"Shapiro-Wilk for {var}, p-value: {sts.shapiro(data_port[var]).pvalue: .10f}")

# If Shapiro significant => data not normally distributed -> they are all not normally distributed

#Create a new column that indicates if G3 is passed or not
data_math['G3 passed'] = data_math['G3'].apply(lambda x: 'Passed' if x >= 10 else 'Failed')
data_port['G3 passed'] = data_port['G3'].apply(lambda x: 'Passed' if x >= 10 else 'Failed')

#Create a new column that converts the grades into a 5-level scale
bins = [0, 10, 12, 14, 16, 21]
labels = ['F', 'D', 'C', 'B', 'A']
data_math['5-level grade'] = pd.cut(data_math['G3'], bins=bins, labels=labels, right=False)
data_port['5-level grade'] = pd.cut(data_port['G3'], bins=bins, labels=labels, right=False)


#Combine the datasets
data_merged = pd.concat([data_math, data_port])

#Define a Random Forest Algorithm
def RF_regressor(data, label):
    # parameter grid for the hyperparameters
    parameter_grid = {'n_estimators': [100, 200, 300, 400],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [10, 20, 30, None],'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4], 'bootstrap': [True, False]}

    # Select Features and Labels
    X = data.drop(["G3", "G3 passed", "5-level grade", "G1", "G2", 'age'], axis=1)  # Features
    y = data[label]  # Label
    X = pd.get_dummies(X, columns=cat_cols, drop_first = True) #one hot encoding

    # Metrics for the results
    metrics = {'Root Mean Squared error': [], 'Mean absolute error': [], 'R^2': []}

    # Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=47)

    #Conduct Cross Validation
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # create Random forest regressor
        global best_n_estimator, best_max_features, best_max_depth, best_min_samples_split,
        best_min_samples_leaf, best_bootstrap
        rf_regressor = RandomForestRegressor(random_state=47)
        #Conduct GridSearch cross validation
        Reg_GS = GridSearchCV(rf_regressor, parameter_grid, cv = 5, verbose=2)
        Reg_GS.fit(X_train, y_train)
        best_n_estimator = Reg_GS.best_estimator_.get_params()['n_estimators']
        best_max_features = Reg_GS.best_estimator_.get_params()['max_features']
        best_max_depth = Reg_GS.best_estimator_.get_params()['max_depth']
        best_min_samples_split = Reg_GS.best_estimator_.get_params()['min_samples_split']
        best_min_samples_leaf = Reg_GS.best_estimator_.get_params()['min_samples_leaf']
        best_bootstrap = Reg_GS.best_estimator_.get_params()['bootstrap']

        #Make predictions
        y_pred = Reg_GS.best_estimator_.predict(X_test)

        #calculate metrics
        global rmse, mae, r2
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        #store metrics
        metrics['Root Mean Squared error'].append(rmse)
        metrics['Mean absolute error'].append(mae)
        metrics['R^2'].append(r2)

    #Calculate mean and std of the metrics
    mean_metrics = {key: np.mean(values) for key, values in metrics.items()}
    std_metrics = {key: np.std(values) for key, values in metrics.items()}

    for key, value in mean_metrics.items():
        print('{}: {:.3f} ± {:.3f}'.format(key.capitalize(), value, std_metrics[key]))



def RF_classifier(data, n_estimators, label):
    # Select Features and Labels
    X = data.drop(["G3", "G3 passed", "5-level grade", "G1", "G2", 'age'], axis=1)  # Features
    y = data[label]  # Label
    X = pd.get_dummies(X, columns=cat_cols, drop_first = True) #one hot encoding

    #Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

    #Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #create Random forest classifier
    rf_classifier = RandomForestClassifier(n_estimators, random_state=47)

    #train model on training data
    rf_classifier.fit(X_train, y_train)

    #Predict for testdata
    y_pred = rf_classifier.predict(X_test)

    #Get performance parameters
    global accuracy, precision, f1, feature_importance_df
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    #Get most important features
    feature_importances = rf_classifier.feature_importances_
    feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

    #CrossValidation

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=47)

    # CV and assessment
    cv_scores = cross_val_score(rf_classifier, X_train, y, cv=kf, scoring='r2')
    print(f"Cross-Validation R^2 Scores: {cv_scores}")
    print(f"Mean CV R^2 Score: {cv_scores.mean()}")
    print(f"Standard Deviation of CV R^2 Score: {cv_scores.std()}")



#Define an overview table for the model results
overview = pd.DataFrame(columns=['Accuracy', 'Precision', 'F1', 'mae', 'rmse', 'r2',
                                 'main feature', 'top 5 features'])
#create a dictionary for the data-frame names
data_titles = {'Math':data_math, 'Portuguese':data_port,'Merged':data_merged}
#create a function that allows to get the name of a dataframe
def get_dataset_name(df):
    for name, dataset in data_titles.items():
        if df.equals(dataset):
            return name
    return None

#conduct several random forest iterations with different labels and different data
for data in [data_math, data_port, data_merged]:
    RF_regressor(data, 'G3')
    overview.loc[f'{get_dataset_name(data)} - {label}', 'rmse'] = rmse
    overview.loc[f'{get_dataset_name(data)} - {label}', 'mae'] = mae
    overview.loc[f'{get_dataset_name(data)} - {label}', 'r2'] = r2
    overview.loc[f'{get_dataset_name(data)} - {label}', 'main feature'] = feature_importance_df.Feature.iloc[0]
    overview.loc[f'{get_dataset_name(data)} - {label}', 'top 5 features'] = feature_importance_df.Feature.head(5).to_list()
    for label in ['G3 passed', '5-level grade']:
        RF_classifier(data, 200, label)
        overview.loc[f'{get_dataset_name(data)} - {label}', 'Accuracy'] = accuracy
        overview.loc[f'{get_dataset_name(data)} - {label}', 'Precision'] = precision
        overview.loc[f'{get_dataset_name(data)} - {label}', 'F1'] = f1
        overview.loc[f'{get_dataset_name(data)} - {label}', 'main feature'] = feature_importance_df.Feature.iloc[0]
        overview.loc[f'{get_dataset_name(data)} - {label}', 'top 5 features'] = feature_importance_df.Feature.head(5).to_list()

overview