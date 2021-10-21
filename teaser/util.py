from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
import scipy.sparse

csr_matrix = scipy.sparse.csr_matrix


#################################
#   Save/load/parse utilities   #
#################################


def parse_interactions(interactions: Path, item_id, user_id, shape_items=0):
    """ Load interactions from csv to csr_matrix. """
    data = pd.read_csv(interactions)

    X = interactions_df_to_csr(data, item_id, user_id, shape_items=shape_items)
    return X


def parse_metadata(metadata: Path, item_id):
    """ Load metadata from csv to csr_matrix. Returns values and labels """
    data = pd.read_csv(metadata).set_index(item_id).sort_index()

    values = data.values.astype(np.int8)
    S = scipy.sparse.csr_matrix(values, dtype=np.int8)
    return S, list(data.columns)


def store_interactions(X: csr_matrix, path: Path, item_id: str, user_id: str):
    """ Write interactions to csv file. """
    rows, cols = X.nonzero()
    df = pd.DataFrame(data={user_id: rows, item_id: cols})
    df.to_csv(path, index=False)


def interactions_df_to_csr(interactions: pd.DataFrame, item_id, user_id, shape_items=0):
    """ Converts a pandas dataframe to user-item csr matrix. """
    max_user = interactions[user_id].max() + 1
    max_item = max(interactions[item_id].max() + 1, shape_items)
    values = np.ones(len(interactions), dtype=np.int8)
    X = scipy.sparse.csr_matrix((values, (interactions[user_id], interactions[item_id])), dtype=np.int8, shape=(max_user, max_item))

    # duplicates are added together -> make binary again
    X[X > 1] = 1

    return X


def remap_ids(*dfs, col):
    """ Maps the ids in a certain column of each dataframe to consecutive integer identifiers.
    This operation is handled inplace.
    """
    keys = set()
    for df in dfs:
        keys |= set(df[col].unique())
    id_mapping = pd.Series(np.arange(len(keys)), index=list(keys))
    for df in dfs:
        df[col] = id_mapping[df[col]].values
    return dfs


#########################
#   Numeric utilities   #
#########################

def rescale(a, lower=-1, upper=1):
    """ Rescale values in an array linearly to a certain range """
    a = a.copy()
    l, u = a.min(), a.max()
    a -= l              # positive with minimum 0
    a /= u - l          # between 0 and 1
    a *= upper - lower  # between 0 and (upper - lower)
    a += lower          # between lower and upper
    return a


def rescale0(a, neg=-1, pos=1):
    """ Rescale values in an array such that zero remains unscaled and
    highest absolute value is assumed as upper and lower bound (stretch negative and positive sides to targets). """
    a = a.copy()
    m = np.abs(a).max()

    if neg == -pos:
        a *= pos / m
    else:
        a[a < 0] *= -neg / m
        a[a > 0] *= pos / m

    return a


def normalize(a):
    """ Make array sum up to one. """
    if np.all(a == 0):
        return a
    return a / a.sum()


def diag_dot(A, B):
    """ Returns diagonal of dot product between A and B. """
    min_outer_size = min(A.shape[0], B.shape[1])

    A = A[:min_outer_size]
    B = B[:, :min_outer_size]
    if scipy.sparse.issparse(B):
        return np.asarray(np.sum(B.T.multiply(A), axis=1)).flatten()
    elif scipy.sparse.issparse(A):
        return np.asarray(np.sum(A.multiply(B.T), axis=1)).flatten()
    else:
        return np.sum(A * B.T, axis=1).flatten()


################################
#   Recommendation utilities   #
################################

def prediction_to_recommendations(predictions: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Takes a row of user-item scores and returns a ranked list of the top_k items along with their scores. """
    if top_k == 0:
        return np.array([]), np.array([])
    recommendations = np.argpartition(predictions, -1-np.arange(top_k))[-top_k:][::-1]
    scores = predictions[recommendations]
    return recommendations, scores

def predictions_to_recommendations(predictions: np.ndarray, top_k: int) -> np.ndarray:
    """ Takes a matrix of user-item scores and returns a ranked list of the top_k items per user. """
    recommendations = np.argpartition(predictions, -1-np.arange(top_k), axis=1)[:, -top_k:][:, ::-1]
    # scores = np.take_along_axis(predictions, recommendations, axis=1)
    return recommendations


def split(X: csr_matrix, test_users: int, perc_history: float, min_interactions: int = 4, seed: int = 42) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
    """ Splits interaction matrix X in three parts: training, val_in and val_out with strong generalization.
    Users in training and validation are disjoint.
    """
    # set seed for reproducability
    np.random.seed(seed)
    users = X.shape[0]
    assert users > test_users, "There should be at least one train user left"

    # pick users with at a certain amount of interactions
    active_users = np.where(X.sum(axis=1) >= min_interactions)[0]
    if len(active_users) < test_users:
        raise ValueError(f"Can't select {test_users} test users. There are only {len(active_users)} users with at least {min_interactions} interactions.")

    test_user_ids = np.random.choice(active_users, test_users, replace=False)
    test_user_mask = np.zeros(users, dtype=bool)
    test_user_mask[test_user_ids] = 1
    train, val = X[~test_user_mask], X[test_user_mask]
    train.eliminate_zeros()

    val_in = val.copy()
    for u in range(val_in.shape[0]):
        items = val[u].nonzero()[1]
        amt_out = int(len(items) * (1 - perc_history))
        amt_out = max(1, amt_out)                   # at least one test item required
        amt_out = min(len(items) - 1, amt_out)      # at least one train item required
        items_out = np.random.choice(items, amt_out, replace=False)

        val_in[u, items_out] = 0

    val_in.eliminate_zeros()

    val_out = val
    val_out[val_in.astype(bool)] = 0
    val_out.eliminate_zeros()

    return train, val_in, val_out
