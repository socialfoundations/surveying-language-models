import math
import numpy as np
import pandas as pd
from tqdm import tqdm

def col2freq(col, n_cats, assign_nan=None, normalize=True, weight=None):
    """
    Inputs a pandas frame and extracts the relative frequency of each choice.
    If assign_nan is an int, then assign its rel. freq. to such choice.
    """
    p = np.zeros(n_cats)

    if weight is None:
        freqs = col.value_counts(dropna=assign_nan is None, normalize=normalize)
    else:
        df = pd.concat([col, weight], axis=1)
        freqs = df.groupby(col.name, dropna=assign_nan is None)[weight.name].sum()
        if normalize:
            freqs = freqs / freqs.sum()

    for choice, prob in freqs.items():
        if not math.isnan(choice):
            p[int(choice) - 1] = prob

    if assign_nan is not None:
        p[assign_nan - 1] += freqs[float('nan')]
    return p

def openai_upper_bound(logp, psum, minlogp):
    """ Since the OpenAI API only gives top-k logits, fill the missing entries with an upper bound """
    not_seen = logp == -np.inf
    n_not_seen = not_seen.sum()
    p = np.exp(logp)
    if n_not_seen > 0:
        pleft = (1. - psum) / n_not_seen
        pmin = min(pleft, np.exp(minlogp))
        p[not_seen] = pmin
    return p / p.sum()

def process_naive(dataset, n_categories, isopenai=False):
    """
    Inputs: pd.DataFrame output by naively filling the form
    Returns: dictionary, (survey question: response distribution)
    """
    distributions = {}
    for _, row in dataset.iterrows():
        var = row['var']
        p = np.array([row[str(i+1)] for i in range(n_categories[var])])
        if isopenai:
            p = openai_upper_bound(p, row['sp'], row['mlogp'])
        distributions[var] = p / p.sum()
    return distributions

def process_adjusted(df, n_cats, isopenai=False):
    pkey = 'logp' if isopenai else 'p'
    c_columns = ['c' + str(i) for i in range(n_cats)]
    p_columns = [pkey + str(i) for i in range(n_cats)]

    # Get the probabilities (in the order that they were presented)
    p = df[p_columns].to_numpy()

    # If openai, upper bound
    if isopenai:  # since it is logps
        sumps = df['sp'].to_numpy()
        minlogps = df['mlogp'].to_numpy()
        p = np.array([openai_upper_bound(p[i], sumps[i], minlogps[i]) for i in range(p.shape[0])])

    choices_p = p.mean(axis=0)

    # Place the probabilities in the choice order, and average
    ids = df[c_columns].to_numpy() - 1  # since 1-indexed
    p = np.array([p[i, ids[i]] for i in range(p.shape[0])])
    p = p.mean(axis=0)
    return choices_p, p

def get_naive_from_adjusted(df, n_cats, isopenai=False):
    pkey = 'logp' if isopenai else 'p'
    c_columns = ['c' + str(i) for i in range(n_cats)]
    p_columns = [pkey + str(i) for i in range(n_cats)]

    # Get the probabilities (in the order that they were presented)
    p = df[p_columns].to_numpy()

    # If openai, upper bound
    if isopenai:  # since it is logps
        sumps = df['sp'].to_numpy()
        minlogps = df['mlogp'].to_numpy()
        p = np.array([openai_upper_bound(p[i], sumps[i], minlogps[i]) for i in range(p.shape[0])])

    choices_p = p.mean(axis=0)

    # Place the probabilities in the choice order, and average
    ids = df[c_columns].to_numpy() - 1  # since 1-indexed
    p = np.array([p[i, ids[i]] for i in range(p.shape[0])])
    p = p.mean(axis=0)
    return choices_p, p

def load_naive_responses(dir, models, n_categories, openai_models=None, ablation=None):
    print("Loading naive responses...")
    variables = n_categories.keys()
    appendix = '_naive.csv'
    if ablation is not None:
        appendix = f'_{ablation}{appendix}'

    naive_responses = {var: {} for var in variables}
    for model in tqdm(models):
        results = pd.read_csv(dir + model + appendix)
        isopenai = openai_models is not None and model in openai_models
        responses = process_naive(results, n_categories, isopenai=isopenai)
        for var, response in responses.items():
            naive_responses[var][model] = response
    return naive_responses

def load_adjusted_responses(dir, models, n_categories, openai_models=None, ablation=None):
    print("Loading adjusted responses...")
    variables = n_categories.keys() if type(n_categories) == dict else n_categories
    ablation = '' if ablation is None else '_' + ablation
    choice_responses = {var: {} for var in variables}
    adjusted_responses = {var: {} for var in variables}
    for model in tqdm(models):
        for var in variables:
            results = pd.read_csv(dir + model + f"{ablation}_{var}.csv")
            isopenai = openai_models is not None and model in openai_models
            n_cats = n_categories[var] if type(n_categories) == dict else results.shape[1] //2
            choices_p, p = process_adjusted(results, n_cats, isopenai=isopenai)
            choice_responses[var][model] = choices_p
            adjusted_responses[var][model] = p
    return choice_responses, adjusted_responses