### LINEAR REGRESSION ###

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os

def perform_linear_regression_and_plot(data, subject, output_folder):
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
    plt.savefig(os.path.join(output_folder, f"LinearRegression_{subject}.png"), dpi=100)
    plt.close()

    # Report the performance metrics
    print(f"Performance Metrics for {subject}:")
    print(f"R2 scores: {r2_scores}")
    print(f"Mean R2 score: {np.mean(r2_scores):.3f}")
    print(f"RMSE scores: {rmse_scores}")
    print(f"Mean RMSE: {np.mean(rmse_scores):.3f}")

# Define output folder
output_folder = "../output"

# Perform linear regression and generate plots for each dataset
perform_linear_regression_and_plot(data_math, "Mathematics", output_folder)
perform_linear_regression_and_plot(data_port, "Portuguese", output_folder)
perform_linear_regression_and_plot(data_merged, "Merged", output_folder)