# Perform the signal test on the model-generated dataset, as well as the actual ACS dataset

from tqdm import tqdm

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

features = [
    'SEX', 'AGER', 'HISPR', 'RAC1PR', 'NATIVITY', 'CIT',
    'SCH', 'SCHLR', 'LANX', 'ENG',
    'HICOV', 'DEAR', 'DEYE',
    'MAR', 'FER', 'GCL', 'MIL',
    'WRK', 'ESR', 'JWTRNS', 'WKL', 'WKWN', 'WKHPR', 'COWR', 'PINCPR'
]

# We construct a binary prediction task for every variable, by considering y=1 if choice >= binary_labels[var]
binary_labels = {
    'SEX': 2,  # predict Male vs Female
    'HISPR': 2,  # predict Hispanic vs Non-Hispanic
    'RAC1PR': 2,  # predict White alone vs Non-White alone
    'NATIVITY': 2,  # predict born in the U.S. vs not
    'CIT': 2,  # predict U.S. Citizen born in the U.S.
    'SCH': 2,  # predict has not attended school in the last 3 months
    'SCHLR': 4,  # predict highest level of education is >= college
    'LANX': 2,  # predict speaks a language other than English at home
    'ENG': 2,  # predict speaks English "very well"
    'MAR': 2,  # predict "now married"
    'WRK': 2,  # predict "worked last week"
    'ESR': 2,  # predict "civilian employed, at work"
    'WKL': 2,  # predict "last worked within the past 12 months"
    'WKWN': 5,  # predict "last year worked 48 to 52 weeks"
    'WKHPR': 4,  # predict "worked 35 or more hours per week"
    'COWR': 2,  # predict "private for-profit company worker"
    'PINCPR': 4,  # predict "income above $52,000"
}

# For some questions, the question may not be asked depending on the answer to some previous question.
# Remove such previous questions from the predictive task, since they influence the target variable deterministically
# due to the survey design rather than how the language model samples the data.
drop_pred = {
    'WRK': ['COWR'],
    'LANX': ['ENG'],
    'SEX': ['FER'],
    'ESR': ['JWTRNS'],
    'WKL': ['WKWN', 'WKHPR', 'COWR'],
    'COWR':['WRK'],
}

# Categorical variables (handled differently for Logistic Regression)
categorical = ['RAC1PR', 'CIT', 'SCH', 'SCHLR', 'MAR', 'MIL', 'ESR', 'JWTRNS', 'COWR']


def build_prediction_task(data, var):
    """ Construct a binary prediction task for the given variable

    Inputs
    ------
    data: pandas DataFrame, the (model-generated) dataset
    var: str, the variable to predict

    Returns
    -------
    X: pandas DataFrame, the features of the prediction task
    Y: pandas Series, the target variable of the prediction task
    """
    # Remove rows for which the target variable is missing
    data = data.dropna(subset=[var])
    # Keep all other missing values, and recode them as 0
    data = data.fillna(0).astype('Int64')

    # Construct the binary prediction task
    Y = (data[var] >= binary_labels[var]).astype('Int64')
    X = data.drop(var, axis=1)

    # Remove additional features if necessary
    if var in drop_pred.keys():
        to_drop = drop_pred[var]
        for v_drop in to_drop:
            if v_drop in X.columns:
                X = X.drop(v_drop, axis=1)

    return X, Y

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


def evaluate_ml_models(data, var, seed):
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

    # Build the prediction task
    X, Y = build_prediction_task(data, var)

    # 20% train-test split
    N = len(X)
    M = int(N * 0.8)
    ids = np.arange(N)
    np.random.shuffle(ids)
    X_train, y_train = X.iloc[ids[:M]], Y.iloc[ids[:M]]
    X_test, y_test = X.iloc[ids[M:]], Y.iloc[ids[M:]]

    # one hot encode for logistic regression
    to_cat = list((set(categorical) & set(X.columns)) - {var})
    X_log = pd.get_dummies(X, columns=to_cat, drop_first=True, dummy_na=True)
    X_train_log, X_test_log = X_log.iloc[ids[:M]], X_log.iloc[ids[M:]]

    accuracies = {}

    # Prevalence of the majority class
    rate_0 = y_test.mean()
    accuracies['constant'] = max(rate_0, 1. - rate_0)

    # Categorical naive bayes
    naive_bayes_model = CategoricalNB()
    accuracies['nb'] = get_test_acc_model(naive_bayes_model, X_train, X_test, y_train, y_test)

    # Logistic regression
    log_reg_model = make_pipeline(StandardScaler(), LogisticRegression())  # otherwise convergence issues
    accuracies['logistic'] = get_test_acc_model(log_reg_model, X_train_log, X_test_log, y_train, y_test)

    # XGBoost
    xgb_model = XGBClassifier()
    accuracies['xgboost'] = get_test_acc_model(xgb_model, X_train, X_test, y_train, y_test)

    return accuracies


def signal_test_all_vars(data, n_seeds):
    results = []
    # For every prediction task...
    for var in tqdm(binary_labels.keys()):
        # Evaluate the performance for 10 different random train-test splits
        for seed in range(n_seeds):
            accuracies = evaluate_ml_models(data, var, seed)
            results.append({'var': var, 'seed': seed, **accuracies})
    return pd.DataFrame(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default=None)  # base dir + model name
    parser.add_argument('--acs_data_file', type=str, default=None)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--n_seeds', type=int, default=10)

    args = parser.parse_args()

    assert ~((args.base_dir is None) and (args.acs_data_file is None)), "Either base_dir or acs_data_file must be specified"

    if args.base_dir is not None:
        model_name = args.base_dir.split('/')[-1]
        print(f"Running signal test for {model_name}...")

        # Load the model generated data (which was generated using 50 seeds)
        dfs = []
        for seed in range(50):
            fname = f"{args.base_dir}_s{seed}.csv"
            dfs.append(pd.read_csv(fname))
        data = pd.concat(dfs, ignore_index=True)
        data = data[features]

        # Perform the signal test
        results = signal_test_all_vars(data, n_seeds=args.n_seeds)

        # Save the results
        results.to_csv(f"{args.save_dir}{model_name}_accuracies.csv", index=False)

    if args.acs_data_file is not None:
        # For the actual ACS data...
        print("Running signal test for ACS data...")
        data = pd.read_csv(args.acs_data_file)
        weights = data['PWGTP']
        data = data[features]  # remove state and weight columns (not sampled by language models)
        data = data.sample(n=100000, weights=weights, random_state=0)  # such that sizes match
        results = signal_test_all_vars(data, n_seeds=args.n_seeds)
        results.to_csv(args.save_dir + "acs_accuracies.csv", index=False)