# Perform the discriminator test on the model-generated dataset, as well as the actual ACS dataset

import os
from tqdm import tqdm

import numpy as np
import pandas as pd

from xgboost import XGBClassifier


features = [
    'SEX', 'AGER', 'HISPR', 'RAC1PR', 'NATIVITY', 'CIT',
    'SCH', 'SCHLR', 'LANX', 'ENG',
    'HICOV', 'DEAR', 'DEYE',
    'MAR', 'FER', 'GCL', 'MIL',
    'WRK', 'ESR', 'JWTRNS', 'WKL', 'WKWN', 'WKHPR', 'COWR', 'PINCPR'
]


def get_test_acc_model(model, X_train, X_test, y_train, y_test):
    """ Train the model on the training data and return the test accuracy.

    Inputs
    ------
    model: sklearn model, the model to train and test
    X_train, X_test: pandas DataFrame, the training and test features
    y_train, y_test: pandas Series, the training and test target variables

    Returns
    -------
    test_acc: float, the test accuracy of the model
    """
    model.fit(X_train, y_train)
    test_acc = (y_test == model.predict(X_test)).mean()
    return test_acc


def evaluate_ml_models(X, Y, seed):
    """ Evaluate the performance of several machine learning models on the prediction task for the given variable.

    Inputs
    ------
    data: pandas DataFrame, the (model-generated) dataset
    var: str, the variable to predict

    Returns
    -------
    accuracies, dict, the test accuracies of each of the ML models evaluated.
    """
    np.random.seed(seed)

    # 20% train-test split
    N = len(X)
    M = int(N * 0.8)
    ids = np.arange(N)
    np.random.shuffle(ids)
    X_train, y_train = X.iloc[ids[:M]], Y.iloc[ids[:M]]
    X_test, y_test = X.iloc[ids[M:]], Y.iloc[ids[M:]]

    accuracies = {}

    # Prevalence of the majority class
    rate_0 = y_test.mean()
    accuracies['constant'] = max(rate_0, 1. - rate_0)

    # XGBoost
    xgb_model = XGBClassifier()
    accuracies['xgboost'] = get_test_acc_model(xgb_model, X_train, X_test, y_train, y_test)

    return accuracies


def signal_test_all_vars(data, census_data, weights, n_seeds):
    data = data.fillna(0).astype('Int64')
    census_data = census_data.fillna(0).astype('Int64')

    results = []
    data['SYNTH'] = 1

    # For every prediction seed...
    for seed in tqdm(range(n_seeds)):
        # Subsample census data
        census_data_sub = census_data.sample(n=len(data), weights=weights, random_state=seed)

        # Create the prediction task
        census_data_sub['SYNTH'] = 0
        all_data = pd.concat([data, census_data_sub])
        all_data = all_data.sample(n=len(all_data), random_state=seed)
        Y = all_data['SYNTH']
        X = all_data.drop(columns=['SYNTH'])

        accuracies = evaluate_ml_models(X, Y, seed)
        results.append({'seed': seed, **accuracies})

    return pd.DataFrame(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True)  # base dir + model name
    parser.add_argument('--acs_data_file', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--n_seeds', type=int, default=10)

    args = parser.parse_args()

    model_name = args.base_dir.split('/')[-1]
    print(f"Running signal test for {model_name}...")

    # Load the ACS data
    print("Loading ACS data...")
    census_data = pd.read_csv(args.acs_data_file)
    weights = census_data['PWGTP']
    census_data = census_data[features]

    # Load the model generated data
    directory = args.base_dir[:args.base_dir.rfind('/')]
    files = os.listdir(directory)
    dfs = []
    for file in files:
        if file.startswith(model_name) and file.endswith('.csv') and file[len(model_name)] == '_':
            dfs.append(pd.read_csv(f"{directory}/{file}"))
    data = pd.concat(dfs, ignore_index=True)
    data = data[features]
    print(f"Loaded {len(data)} rows of model-generated data.")

    # Perform the signal test
    results = signal_test_all_vars(data, census_data, weights, n_seeds=args.n_seeds)

    # Save the results
    results.to_csv(f"{args.save_dir}{model_name}_accuracies.csv", index=False)

    print("Done!")