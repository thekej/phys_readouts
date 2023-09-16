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

list_include = {'collide':
['pilot_it2_collision_assorted_targets_tdw_1_dis_1_occ',
'pilot_it2_collision_simple_box',
'pilot_it2_collision_assorted_targets_box',
'pilot_it2_collision_tiny_ball_box',
'pilot_it2_collision_tiny_ball_tdw_1_dis_1_occ',],

'contain':
    ['pilot-containment-multi-bowl',
'pilot-containment-bowl',
'pilot-containment-vase_torus',
'pilot-containment-cone-plate',],

'dominoes':
['pilot_dominoes_0mid_d3chairs_o1plants_tdwroom',
'pilot_dominoes_1mid_J025R45_o1full_tdwroom',
'pilot_dominoes_4midRM1_tdwroom',
'pilot_dominoes_4midRM1_boxroom_2',
'pilot_dominoes_SJ020_d3chairs_o1plants_tdwroom',
'pilot_dominoes_default_boxroom'],

'drop':
['pilot_it2_drop_all_bowls_box',
'pilot_it2_drop_all_bowls_tdw_1_dis_1_occ',
'pilot_it2_drop_simple_tdw_1_dis_1_occ',
'pilot_it2_drop_simple_box',],

'link':
['pilot_linking_nl1-8_mg000_aCyl_bCyl_tdwroom1',
'pilot_linking_nl1-8_mg000_aCyl_bCyl_tdwroom_small_rings',
'pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom1',
'pilot_linking_nl1-8_mg000_aNone_bCyl_tdwroom_small_rings',
'pilot_linking_nl6_aNone_bCone_occ1_dis1_boxroom',],

'roll':
['pilot_it2_rollingSliding_simple_ramp_box_small_zone',
'pilot_it2_rollingSliding_simple_ledge_box_sphere_small_zone',
'pilot_it2_rollingSliding_simple_ramp_tdw_1_dis_1_occ_small_zone',
'pilot_it2_rollingSliding_simple_collision_box',
'pilot_it2_rollingSliding_simple_ledge_box',
'pilot_it2_rollingSliding_simple_collision_tdw_1_dis_1_occ',
'pilot_it2_rollingSliding_simple_ledge_tdw_1_dis_1_occ_sphere_small_zone'],

'support':
['pilot_towers_nb2_fr015_SJ010_mono0_dis0_occ0_tdwroom_unstable',
'pilot_towers_nb4_fr015_SJ000_gr01_mono0_dis1_occ1_tdwroom_unstable',
'pilot_towers_nb2_fr015_SJ010_mono0_dis0_occ0_tdwroom_stable',
'pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable'] }



def load_hdf5(file_path, dataset_name, indices=None, mean=False):

    # breakpoint()

    # file_path = file_path.split('.')[-2] + '_temp.hdf5'

    with h5py.File(file_path, 'r') as file:
        if indices is None:
            dataset = file[dataset_name][...]
        else:
            dataset = file[dataset_name][sorted(indices)]

    #save stimulus map here if it doesn't exist.
    if mean:
        if dataset_name != 'label':
            # dataset = dataset.mean(axis=-2)
            dataset = dataset.reshape(dataset.shape[0], 2, 64, 64, 256)
            dataset = dataset[:, :, ::2, ::2, :]

    # breakpoint()

    return dataset

def balance_data(X, y):
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    return X_resampled, y_resampled

# def save_temp_hdf5(file_path):
#     with h5py.File(file_path, 'r') as file:
#         dataset = file['features'][:]
#         label = file['label'][:]
#         filenames = file['filenames'][:]
#
#         with h5py.File(file_path.split('.')[-2] + '_temp.hdf5', 'w') as f:
#             f.create_dataset('features', data=dataset)
#             f.create_dataset('label', data=label)

def load_save_hdf5(file_path, include=True):

    #save this in same folder as hdf5 file

    # breakpoint()

    all_scenarios = {'collide': [], 'drop': [], 'support': [], 'link': [], 'roll': [], 'contain': [], 'dominoes': []}

    # all_data_scenarios, all_labels_scenarios = {'collide': [], 'drop': [], 'support': [], 'link': [], 'roll': [], 'contain': [], 'dominoes': []}, {'collide': [], 'drop': [], 'support': [], 'link': [], 'roll': [], 'contain': [], 'dominoes': []}

    with h5py.File(file_path, 'r') as file:
        dataset = file['features'][:]
        label = file['label'][:]
        filenames = file['filenames'][:]
        # breakpoint()
        dataset = dataset.mean(axis=-2)

        dataset = dataset.reshape(dataset.shape[0], -1)#[:, :10]

        # dataset

        all_data, all_labels = [], []
        ct_ = 0
        for scenario in list_include.keys():
            X_imb, y_imb = [], []
            for ct, x in enumerate(dataset):
                cfg = '_'.join(filenames[ct].decode('utf-8').split('/')[-1].split('.')[0].split('_')[:-1])
                if include:
                    if cfg in list_include[scenario]:
                        X_imb.append(x)
                        y_imb.append(label[ct])
                        ct_ += 1
                        all_scenarios[scenario].append(ct_-1)
                else:
                    if (cfg not in list_include[scenario]) and (scenario + '_all_movies' in filenames[ct].decode('utf-8')):
                        X_imb.append(x)
                        y_imb.append(label[ct])
                        ct_ += 1
                        all_scenarios[scenario].append(ct_-1)

            # breakpoint(
            try:
                X_imb = np.stack(X_imb, axis=0)
                y_imb = np.array(y_imb)
            except:
                breakpoint()

            if include:
                X_imb, y_imb = balance_data(X_imb, y_imb)

            all_data.append(X_imb)
            all_labels.append(y_imb)

            # breakpoint()
        data_path = file_path.split('.')[-2] + '_temp.hdf5'
        with h5py.File(data_path, 'w') as f:
            f.create_dataset('features', data=np.concatenate(all_data, axis=0))
            f.create_dataset('label', data=np.concatenate(all_labels, axis=0))

        # Saving the dictionary as JSON data in a file
        json_path = file_path.split('.')[-2] + '_temp.json'
        with open(json_path, 'w') as json_file:
            json.dump(all_scenarios, json_file)

        #save stimulus map: filename -> index for csv

    return data_path, json_path


def one_scenario_eval(args, model, data, target, scenario, result, indices):
    with open(args.test_scenario_map, 'r') as f:
        stimulus_ = json.load(f)
        stimulus = {v: k for k, v in stimulus_.items()}
        
    accuracy = model.score(data, target)
    result[scenario+'_test'] = accuracy
    print(f"Accuracy on %s test data (%d data points): {accuracy:.4f}"%(scenario, target.shape[0]))
    # breakpoint()
    probs = model.predict_proba(data)
    preds = model.predict(data)
    results = [['Readout Train Data', 'Readout Test Data', 'Train Accuracy', 
              'Test Accuracy', 'Readout Type', 'Predicted Prob_false', 
              'Predicted Prob_true', 'Predicted Outcome', 'Actual Outcome', 
              'Stimulus Name']]
    for i in range(target.shape[0]):
        entry = [scenario, scenario, float(result['train']), float(result[scenario+'_test']),
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
    result[scenario+'_test'] = accuracy
    print(f"Accuracy on %s test data (%d data points): {accuracy:.4f}"%(scenario, target.shape[0]))
    # breakpoint()
    probs = model.predict_proba(data)
    preds = model.predict(data)
    for i in range(target.shape[0]):
        entry = ['all', scenario, float(result['train']), float(result[scenario+'_test']),
                 args.scenario_name, float(probs[i][0]), float(probs[i][1]), int(preds[i]),
                 int(target[i]), stimulus[indices[i]]]
        results.append(entry)
    return result, results
    

def test_model(model, test_data, test_label, args, result):
    with open(args.test_scenario_indices, 'r') as f:
        test_data = test_data.reshape(test_data.shape[0], -1)
        accuracy = model.score(test_data, test_label)
        print(f"Accuracy on full test data: {accuracy:.4f}")
        result['full_test'] = accuracy
        scenarios_indices = json.load(f)

        # breakpoint()
        
        results = [['Readout Train Data', 'Readout Test Data', 'Train Accuracy', 
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
                result[sc+'_test'] = accuracy
                print(f"Accuracy on %s test data: {accuracy:.4f}"%sc)
        if args.one_scenario == 'all':
            filename = args.data_type+'_'+ args.model_type + '_' +args.scenario_name+ \
            '_only_'+str(args.one_scenario)+'_results.csv'
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

    if args.balance_readout:
        data_path, json_path = load_save_hdf5(args.data_path, include=True)
        args.data_path = data_path
        args.train_scenario_indices = json_path

        data_path, json_path = load_save_hdf5(args.test_path, include=False)
        args.test_path = data_path
        args.test_scenario_indices = json_path


    # breakpoint()

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
        scenario_feature = 'features_mid'
    else:
        scenario_feature = args.scenario
    X = load_hdf5(args.data_path, scenario_feature, indices, mean=args.mean)

    # breakpoint()

    if args.scenario_name in ['observed_gamma', 'observed_beta', 'observed_teco_h',  'observed_mcvd_ucf']:
        X = X[:, 0]
    elif args.scenario_name in ['observed_fitvid', 'observed_ego4d_fitvid']:
        print(X.shape)
        X = X[:, :7]
    elif 'teco' in args.scenario_name and args.scenario_name != 'observed_teco_h':
        X = X[:, 1:14]

    if args.data_type == 'bbnet_4x4':
        X = X[:, :, ::2]

    if '3d' in args.data_type:
        X = X[:, -1, :, :11]
        # breakpoint()

    X = X.reshape(X.shape[0], -1)
    y = load_hdf5(args.data_path, 'label', indices)
    # X, y = balance_data(X, y)
    print(X.shape)


    # Define the hyperparameter grid to search
    print('Load model')
    if args.model_type == 'logistic':
        if args.ocp:
            param_grid = {'clf__C': np.array([1e-5, 0.01, 0.1, 1]), 'clf__penalty': ['l2']}  #
        else:
            param_grid = {'clf__C': np.array([1e-5, 0.01, 0.1, 1, 5, 10, 20]), 'clf__penalty': ['l2']}#  np.logspace(-1, 3, 5), 'clf__penalty': ['l2']}
        # param_grid = {'clf__C': np.array([1e-5, 0.01, 0.1]),
        #               'clf__penalty': ['l2']}  # np.logspace(-1, 3, 5), 'clf__penalty': ['l2']}
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
    test_data = load_hdf5(args.test_path, scenario_feature, mean=args.mean)
    if args.scenario_name in ['observed_gamma', 'observed_beta', 'observed_mcvd_ucf' , 'observed_teco_h']:
        test_data = test_data[:, 0]
    elif args.scenario_name in ['observed_fitvid', 'observed_ego4d_fitvid']:
        test_data = test_data[:, :7]
    elif 'teco' in args.scenario_name and args.scenario_name != 'observed_teco_h':
        test_data = test_data[:, 1:14]

    if args.data_type == 'bbnet_4x4':
        test_data = test_data[:, :, ::2]

    if '3d' in args.data_type:
        test_data = test_data[:, -1, :, :11]

    test_label = load_hdf5(args.test_path, 'label')
    result = test_model(grid_search, test_data, test_label, args, result)
    
    filename = args.data_type+'_'+ args.model_type + '_' +args.scenario_name+'_exclude_'+str(args.all_but_one)+'_results.json'     

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
    
    # data params
    parser.add_argument('--data-path', type=str, help='The path to the h5 file')
    parser.add_argument('--data-type', type=str, help='The path to the h5 file')
    parser.add_argument('--test-path', type=str, help='The path to the test file')
    parser.add_argument('--scenario', type=str, default='features')
    parser.add_argument('--scenario-name', type=str, default='features')    
    parser.add_argument('--all-but-one', type=str, default=None,
                        choices=['collide', 'dominoes', 'link', 'support',
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

    #save trur arg
    parser.add_argument('--balance_readout', action='store_true', default=False, help='balance dataset & use unique readouts for training')
    parser.add_argument('--ocp', action='store_true', default=False,
                        help='reg')

    parser.add_argument('--mean', action='store_true', default=False,
                        help='reg')
    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()
