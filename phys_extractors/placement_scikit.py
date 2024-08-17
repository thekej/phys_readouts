import argparse
import csv
import joblib
import json
import h5py
import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


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
    if args.data_type == 'mcvd':
        scenario_feature = 'features'
    else:
        scenario_feature = args.scenario
    X = load_hdf5(args.data_path, scenario_feature)
    if args.scenario_name in ['observed_gamma', 'observed_beta', 'observed_teco_h',  'observed_mcvd_ucf']:
        X = X[:, 0]
    elif args.scenario_name in ['observed_fitvid', 'observed_ego4d_fitvid']:
        print(X.shape)
        X = X[:, :7]
    elif 'teco' in args.scenario_name and args.scenario_name != 'observed_teco_h':
        X = X[:, 1:14]
    X = X.reshape(X.shape[0], -1)
    y = load_hdf5(args.data_path, 'contacts')
    y = y.reshape(-1, 2)
    print(X.shape)


    # Define the hyperparameter grid to search
    print('Load model')
    
    param_grid = {'clf__C': np.array([args.penalty]), 'clf__penalty': ['l2']}
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    # For the 'x' coordinate
    # For the 'x' coordinate
    model_x = LogisticRegression(C=args.penalty, penalty='l2', max_iter=2000, verbose=1)
    pipeline_x = Pipeline([('scaler', StandardScaler()), ('clf', model_x)])

    # For the 'y' coordinate
    model_y = LogisticRegression(C=args.penalty, penalty='l2', max_iter=2000, verbose=1)
    pipeline_y = Pipeline([('scaler', StandardScaler()), ('clf', model_y)])

    #grid_search_x = GridSearchCV(pipeline_x, param_grid, cv=stratified_kfold, verbose=3)

    # For the 'y' coordinate
    #model_y = LogisticRegression(max_iter=200)
    #pipeline_y = Pipeline([('scaler', StandardScaler()), ('clf', model_y)])
    #grid_search_y = GridSearchCV(pipeline_y, param_grid, cv=stratified_kfold, verbose=3)
    print(y.shape, )#[:, 0])
    # Train for x and y coordinates
    pipeline_x.fit(X, y[:, 0])
    pipeline_y.fit(X, y[:, 1])

    # Print the best hyperparameters found by the grid search
    #print(f"Best hyperparameters: {grid_search.best_params_}")
    #result = grid_search.best_params_
    
    # Evaluate the model on the full data
    preds_x = pipeline_x.predict(X)
    preds_y = pipeline_y.predict(X)

    true_coords = np.array(list(zip(y[:, 0], y[:, 1])))
    preds_coords = np.array(list(zip(preds_x, preds_y)))

    joint_mae = mean_absolute_error(true_coords, preds_coords)
    joint_mse = mean_squared_error(true_coords, preds_coords)
    joint_r2 = r2_score(true_coords, preds_coords)
    
    correct_predictions = np.sum((y[:, 0] == preds_x) & (y[:, 1] == preds_y))
    total_predictions = len(true_coords)  # assuming true_x and true_y have the same length
    accuracy = correct_predictions / total_predictions

    print(f"Accuracy on train data: {accuracy:.4f} \n",
          f"MAE on train data: {joint_mae:.6f} \n",
         f"MSE on train data: {joint_mse:.6f} \n",
         f"R2 on train data: {joint_r2:.6f}",)
#    result['train'] = f"{accuracy:.4f}"
    
    #
    test_data = load_hdf5(args.test_path, scenario_feature)
    test_data = test_data.reshape(test_data.shape[0], -1)
    if args.scenario_name in ['observed_gamma', 'observed_beta', 'observed_mcvd_ucf' , 'observed_teco_h']:
        test_data = test_data[:, 0]
    elif args.scenario_name in ['observed_fitvid', 'observed_ego4d_fitvid']:
        test_data = test_data[:, :7]
    elif 'teco' in args.scenario_name and args.scenario_name != 'observed_teco_h':
        test_data = test_data[:, 1:14]
    test_label = load_hdf5(args.test_path, 'contacts')
    test_label = test_label.reshape(-1, 2)
    #result = test_model(grid_search, test_data, test_label, args, result)
    
    #filename = args.data_type+'_'+ args.model_type + '_' +args.scenario_name+'_exclude_'+str(args.all_but_one)+f'_penalty_{args.penalty}_results.json'     
    
    # Evaluate the model on the full data
    preds_x = pipeline_x.predict(test_data)
    preds_y = pipeline_y.predict(test_data)

    true_coords = np.array(list(zip(test_label[:, 0], test_label[:, 1])))
    preds_coords = np.array(list(zip(preds_x, preds_y)))

    joint_mae = mean_absolute_error(true_coords, preds_coords)
    joint_mse = mean_squared_error(true_coords, preds_coords)
    joint_r2 = r2_score(true_coords, preds_coords)
    
    correct_predictions = np.sum((test_label[:, 0] == preds_x) & (test_label[:, 1] == preds_y))
    total_predictions = len(true_coords)  # assuming true_x and true_y have the same length
    accuracy = correct_predictions / total_predictions
    
    print(f"Accuracy on test data: {accuracy:.4f} \n",
          f"MAE on test data: {joint_mae:.6f} \n",
          f"MSE on test data: {joint_mse:.6f} \n",
          f"R2 on test data: {joint_r2:.6f}",)
    
    #result = {key: str(value) for key, value in result.items()}

    #with open(filename, 'w') as f:
    #    json.dump(result, f)


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate a logistic regression model')
    parser.add_argument('--model-path', type=str, default='best_model.joblib', help='path to save or load the best model')
    parser.add_argument('--penalty', type=float, default=1, help='random seed for reproducibility')
    parser.add_argument('--random-state', type=int, default=42, help='random seed for reproducibility')

    # data params
    parser.add_argument('--data-path', type=str, help='The path to the h5 file')
    parser.add_argument('--data-type', type=str, help='The path to the h5 file')
    parser.add_argument('--test-path', type=str, help='The path to the test file')
    parser.add_argument('--scenario', type=str, default='features')
    parser.add_argument('--scenario-name', type=str, default='features')
    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()
