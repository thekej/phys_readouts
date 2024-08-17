import argparse
import csv
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

def one_scenario_eval(args, model, data, target, scenario, result, indices):
    with open(args.test_scenario_map, 'r') as f:
        stimulus_ = json.load(f)
        stimulus = {v: k for k, v in stimulus_.items()}
        
    accuracy = model.score(data, target)
    result[scenario+'_test'] = f"{accuracy:.4f}"
    print(f"Accuracy on %s test data (%d data points): {accuracy:.4f}"%(scenario, target.shape[0]))
    probs = model.predict_proba(data)
    preds = model.predict(data)
    results = [['Readout Train Data', 'Readout Test Data', 'Train Accuracy', 
              'Test Accuracy', 'Readout Type', 'Predicted Prob_false', 
              'Predicted Prob_true', 'Predicted Outcome', 'Actual Outcome', 
              'Stimulus Name']]
    for i in range(target.shape[0]):
        entry = [scenario, args.data_type, scenario, float(result['train']), float(result[scenario+'_test']),
                 args.scenario_name, float(probs[i][0]), float(probs[i][1]), int(preds[i]),
                 int(target[i]), stimulus[indices[i]]]
        results.append(entry)
    filename = args.data_type+'_'+ args.model_type + '_' +args.scenario_name+'_only_'+str(args.one_scenario)+'_results.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results)
    return result


def all_scenario_eval(args, model, data, target, scenario, result, indices, results):
    with open(args.test_scenario_map, 'r') as f:
        stimulus_ = json.load(f)
        stimulus = {v: k for k, v in stimulus_.items()}
        
    accuracy = model.score(data, target)
    result[scenario+'_test'] = f"{accuracy:.4f}"
    print(f"Accuracy on %s test data (%d data points): {accuracy:.4f}"%(scenario, target.shape[0]))
    probs = model.predict_proba(data)
    preds = model.predict(data)
    for i in range(target.shape[0]):
        entry = ['all', args.data_type, scenario, float(result['train']), float(result[scenario+'_test']),
                 args.scenario_name, float(probs[i][0]), float(probs[i][1]), int(preds[i]),
                 int(target[i]), stimulus[indices[i]]]
        results.append(entry)
    return result, results
    

def test_model(model, test_data, test_label, args, result):
    with open(args.test_scenario_indices, 'r') as f:
        test_data = test_data.reshape(test_data.shape[0], -1)
        accuracy = model.score(test_data, test_label)
        print(f"Accuracy on full test data: {accuracy:.4f}")
        result['full_test'] = f"{accuracy:.4f}"
        scenarios_indices = json.load(f)
        
        results = [['Readout Train Data', 'Model', 'Readout Test Data', 'Train Accuracy', 
              'Test Accuracy', 'Readout Type', 'Predicted Prob_false', 
              'Predicted Prob_true', 'Predicted Outcome', 'Actual Outcome', 
              'Stimulus Name']]
        
        for sc in scenarios_indices.keys():
            if args.one_scenario is not None and args.one_scenario != 'all':
                ind = sorted(scenarios_indices[args.one_scenario])
                data, target = test_data[ind], test_label[ind]
                result = one_scenario_eval(args, model, data, target, 
                                  args.one_scenario, result, ind)
                break
            elif args.one_scenario == 'all':
                ind = sorted(scenarios_indices[sc])
                data, target = test_data[ind], test_label[ind]
                result, results = all_scenario_eval(args, model, data, target, 
                                  sc, result, ind,
                                             results)
            else:
                ind = sorted(scenarios_indices[sc])
                accuracy = model.score(test_data[ind], test_label[ind])
                result[sc+'_test'] = f"{accuracy:.4f}"
                print(f"Accuracy on %s test data: {accuracy:.4f}"%sc)
        filename = args.data_type+'_'+ args.model_type + '_' +args.scenario_name+ \
            '_only_'+str(args.one_scenario)+'_exclude_'+str(args.all_but_one)+'_results.csv'
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(results)
    return result
            
def get_indices(scenarios_indices):
    indices = []
    for k in scenarios_indices.keys():
        indices += scenarios_indices[k]
    return indices

def train(args):
    indices = None
    if args.balanced_indices is not None:
        with open(args.balanced_indices, 'r') as f:
            indices = json.load(f)
        
    # account for all but one train protocol
    print('Load data')
    if args.all_but_one is not None:
        with open(args.train_scenario_indices, 'r') as f:
            scenarios_indices = json.load(f)
            if indices is None:
                indices = get_indices(scenarios_indices)
            banned_scenario = scenarios_indices[args.all_but_one]
            indices = list(set(indices) - set(banned_scenario))
            print('Removing %d datapoints from the %s scenario'%(len(banned_scenario),
                                                                 args.all_but_one))
    if args.one_scenario is not None and args.one_scenario != 'all':
        with open(args.train_scenario_indices, 'r') as f:
            scenarios_indices = json.load(f)
            indices = scenarios_indices[args.one_scenario]
            print('Experimenting on %d datapoints from the %s scenario'%(len(indices),
                                                                 args.one_scenario))
            
    if args.data_type == 'mcvd':
        scenario_feature = 'features'
    else:
        scenario_feature = args.scenario
    X = load_hdf5(args.data_path, scenario_feature, indices)
    if args.scenario_name in ['observed_gamma', 'observed_beta', 'observed_teco_h',  'observed_mcvd_ucf']:
        X = X[:, 0]
    elif args.scenario_name in ['observed_fitvid', 'observed_ego4d_fitvid']:
        print(X.shape)
        X = X[:, :7]
    elif 'teco' in args.scenario_name and args.scenario_name != 'observed_teco_h':
        X = X[:, 1:14]
    X = X.reshape(X.shape[0], -1)
    y = load_hdf5(args.data_path, 'label', indices)
    print(X.shape)


    # Define the hyperparameter grid to search
    print('Load model')
    if args.model_type == 'logistic':
        param_grid = {'clf__C': np.array([args.penalty]), 'clf__penalty': ['l2']}#  np.logspace(-1, 3, 5), 'clf__penalty': ['l2']}
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
    stratified_kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=args.random_state)
    grid_search = GridSearchCV(pipeline, param_grid, cv=stratified_kfold, verbose=3)
    grid_search.fit(X, y)
    print(time.time()-t)
    # Print the best hyperparameters found by the grid search
    print(f"Best hyperparameters: {grid_search.best_params_}")
    result = grid_search.best_params_
    
    # Evaluate the model on the full data
    accuracy = grid_search.score(X, y)
    print(f"Accuracy on train data: {accuracy:.4f}")
    result['train'] = f"{accuracy:.4f}"
    
    #
    test_data = load_hdf5(args.test_path, scenario_feature)
    if args.scenario_name in ['observed_gamma', 'observed_beta', 'observed_mcvd_ucf' , 'observed_teco_h']:
        test_data = test_data[:, 0]
    elif args.scenario_name in ['observed_fitvid', 'observed_ego4d_fitvid']:
        test_data = test_data[:, :7]
    elif 'teco' in args.scenario_name and args.scenario_name != 'observed_teco_h':
        test_data = test_data[:, 1:14]
    test_label = load_hdf5(args.test_path, 'label')
    result = test_model(grid_search, test_data, test_label, args, result)
    
    filename = args.data_type+'_'+ args.model_type + '_' +args.scenario_name+'_exclude_'+str(args.all_but_one)+f'_penalty_{args.penalty}_results.json'     
    print(result)
    result = {key: str(value) for key, value in result.items()}

    with open(filename, 'w') as f:
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
    parser.add_argument('--penalty', type=float, default=1, help='random seed for reproducibility')
    
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
    parser.add_argument('--one-scenario', type=str, default=None,
                        choices=['collision', 'domino', 'link', 'towers', 
                                 'contain', 'drop', 'roll', 'all'],
                        help='in case of all-but-one scenario')
    parser.add_argument('--balanced-indices', type=str, default=None, 
                        help='path for scenario mapping')
    parser.add_argument('--train-scenario-indices', type=str, required=True, 
                        help='path for scenario mapping')
    parser.add_argument('--test-scenario-indices', type=str, required=True, 
                        help='path for scenario mapping')
    parser.add_argument('--test-scenario-map', type=str, required=True, 
                        help='path for scenario mapping')
    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()
