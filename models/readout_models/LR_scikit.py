import argparse
import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def train(args):
    # Generate a toy dataset
    X, y = make_classification(n_samples=1000, random_state=args.random_state)

    # Define the hyperparameter grid to search
    param_grid = {'clf__C': np.logspace(-3, 3, 7), 'clf__penalty': ['l1', 'l2']}

    # Create the pipeline with a logistic regression model and a scaler
    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=100))])

    # Perform grid search with stratified k-fold cross-validation
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    grid_search = GridSearchCV(pipeline, param_grid, cv=stratified_kfold)
    grid_search.fit(X, y)

    # Print the best hyperparameters found by the grid search
    print(f"Best hyperparameters: {grid_search.best_params_}")

    # Evaluate the model on the full data
    accuracy = grid_search.score(X, y)
    print(f"Accuracy on full data: {accuracy:.2f}")

    # Save the best model for later use
    joblib.dump(grid_search.best_estimator_, args.model_path)

def evaluate(args):
    # Load the best model from disk
    best_model = joblib.load(args.model_path)

    # Generate new data
    new_X, new_y = make_classification(n_samples=100, random_state=args.random_state)

    # Evaluate the best model on new data
    accuracy = best_model.score(new_X, new_y)
    print(f"Accuracy on new data: {accuracy:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate a logistic regression model')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], help='mode (train or evaluate)')
    parser.add_argument('--model-path', type=str, default='best_model.joblib', help='path to save or load the best model')
    parser.add_argument('--random-state', type=int, default=42, help='random seed for reproducibility')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)

if __name__ == '__main__':
    main()
