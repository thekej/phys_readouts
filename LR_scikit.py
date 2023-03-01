import argparse
import joblib
import json
import h5py
import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC


def load_hdf5(file_path, dataset_name, indices=None):
    with h5py.File(file_path, 'r') as file:
        if indices is None:
            dataset = file[dataset_name][...]
        else:
            dataset = file[dataset_name][sorted(indices)]
    return dataset

def test_model(grid_search, test_data, test_label, args, result):
    with open(args.test_scenario_map, 'r') as f:
        test_data = test_data.reshape(test_data.shape[0], -1)
        accuracy = grid_search.score(test_data, test_label)
        print(f"Accuracy on full test data: {accuracy:.4f}")
        result['full_test'] = accuracy
        scenarios_indices = json.load(f)
        for sc in scenarios_indices.keys():
            ind = sorted(scenarios_indices[sc])
            accuracy = grid_search.score(test_data[ind], test_label[ind])
            result[sc+'_test'] = accuracy
            print(f"Accuracy on %s test data: {accuracy:.4f}"%sc)
    return result
            

def train(args):
    with open(args.balanced_indices, 'r') as f:
        indices = json.load(f)
        
    # account for all but one train protocol
    print('Load data')
    if args.all_but_one is not None:
        with open(args.train_scenario_map, 'r') as f:
            scenarios_indices = json.load(f)
            banned_scenario = scenarios_indices[args.all_but_one]
            indices = list(set(indices) - set(banned_scenario))
            print('Removing %d datapoints from the %s scenario'%(len(banned_scenario),
                                                                 args.all_but_one))
    if args.data_type == 'mcvd':
        scenario_feature = 'features'
    else:
        scenario_feature = args.scenario
    X = load_hdf5(args.data_path, scenario_feature, indices)
    if args.scenario_name in ['observed_gamma', 'observed_beta']:
        X = X[:, 0]
    X = X.reshape(X.shape[0], -1)
    y = load_hdf5(args.data_path, 'label', indices)
    print(X.shape)


    # Define the hyperparameter grid to search
    print('Load model')
    if args.model_type == 'logistic':
        param_grid = {'clf__C': np.logspace(-3, 3, 7), 'clf__penalty': ['l2']}
        model = LogisticRegression(max_iter=20000)
    elif args.model_type == 'svc':
        param_grid = {'clf__C': np.logspace(-3, 0, 4), 'clf__loss': ['hinge']}
        model = LinearSVC(max_iter=10000)

    # Create the pipeline with a logistic regression model and a scaler
    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', model)])

    # Perform grid search with stratified k-fold cross-validation
    print('Grid Search with KFold')
    import time
    t = time.time()
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    grid_search = GridSearchCV(pipeline, param_grid, cv=stratified_kfold, verbose=3)
    grid_search.fit(X, y)
    print(time.time()-t)
    # Print the best hyperparameters found by the grid search
    print(f"Best hyperparameters: {grid_search.best_params_}")
    result = grid_search.best_params_
    
    # Evaluate the model on the full data
    accuracy = grid_search.score(X, y)
    print(f"Accuracy on train data: {accuracy:.4f}")
    result['train'] = accuracy
    
    #
    test_data = load_hdf5(args.test_path, scenario_feature)
    if args.scenario_name in ['observed_gamma', 'observed_beta']:
        test_data = test_data[:, 0]
    test_label = load_hdf5(args.test_path, 'label')
    result = test_model(grid_search, test_data, test_label, args, result)
    with open(args.data_type+'_'+ args.model_type + '_' +args.scenario_name+'_exclude_'+str(args.all_but_one)+'_results.json', 'w') as f:
        json.dump(result, f)

    # Save the best model for later use
    joblib.dump(grid_search.best_estimator_, args.model_path)

def evaluate(args):
    # Load the best model from disk
    best_model = joblib.load(args.model_path)

    # Load the test set
    test_X, test_y = load_hdf5(args.test_data_path)

    # Evaluate the best model on the test set
    accuracy = best_model.score(test_X, test_y)
    print(f"Accuracy on test set: {accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate a logistic regression model')
    parser.add_argument('--model-type', type=str, choices=['logistic', 'svc'], help='type of model to train (logistic or svc)')
    parser.add_argument('--model-path', type=str, default='best_model.joblib', help='path to save or load the best model')
    parser.add_argument('--random-state', type=int, default=42, help='random seed for reproducibility')
    
    # data params
    parser.add_argument('--data-path', type=str, help='The path to the h5 file')
    parser.add_argument('--data-type', type=str, help='The path to the h5 file')
    parser.add_argument('--test-path', type=str, help='The path to the test file')
    parser.add_argument('--scenario', type=str, default='features')
    parser.add_argument('--scenario-name', type=str, default='features')    
    parser.add_argument('--all-but-one', type=str, default=None,
                        choices=['collision', 'domino', 'link', 'towers', 
                                 'contain', 'drop', 'roll'],
                        help='in case of all-but-one scenario')
    parser.add_argument('--balanced-indices', type=str, required=True, 
                        help='path for scenario mapping')
    parser.add_argument('--train-scenario-map', type=str, required=True, 
                        help='path for scenario mapping')
    parser.add_argument('--test-scenario-map', type=str, required=True, 
                        help='path for scenario mapping')
    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()