from collections.abc import Iterable

import numpy as np
import scipy.sparse
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

import teaser.util as util


def recall_k(predictions: np.ndarray, Xval_out: scipy.sparse.csr_matrix, top_k):
    """ Implements a stratified Recall@k that calculates per user:
        #correct / min(k, #user_val_items)
    """
    recommendations = util.predictions_to_recommendations(predictions, top_k=top_k)
    Xval_out = Xval_out[:recommendations.shape[0]]
    hits = np.take_along_axis(Xval_out, recommendations, axis=1)
    total_hits = np.asarray(hits.sum(axis=1)).flatten()
    best_possible = np.asarray(Xval_out.sum(axis=1)).flatten()
    best_possible[best_possible < 1] = 1
    best_possible[best_possible > top_k] = top_k
    recall_scores = total_hits / best_possible
    return recall_scores


def ndcg_k(predictions: np.ndarray, Xval_out: scipy.sparse.csr_matrix, top_k):
    """ Implements normalized discounted cumulative gain.
    """
    recommendations = util.predictions_to_recommendations(predictions, top_k=top_k)
    Xval_out = Xval_out[:recommendations.shape[0]]

    hits = np.take_along_axis(Xval_out, recommendations, axis=1)
    ranks = np.arange(top_k) + 1
    hit_ranks = hits.multiply(ranks)

    hit_ranks.data = 1. / np.log2(hit_ranks.data + 1)
    dcg = np.asarray(hit_ranks.sum(axis=1)).flatten()

    hist_len = Xval_out.getnnz(axis=1).astype(np.int32)
    hist_len[hist_len > top_k] = top_k
    discount_template = 1. / np.log2(np.arange(2, top_k + 2))
    idcg = np.array([(discount_template[:n]).sum() for n in hist_len])

    ndcg = dcg / idcg

    # If we divide 0 by 0 -> set to 0 instead of nan
    ndcg[dcg == 0] = 0

    return ndcg


def eval(alg, Xval_in, Xval_out):
    predictions = alg.predict_all(Xval_in, retarget=False)
    return report_metrics(predictions, Xval_out)


def report_metrics(predictions, Xval_out):
    """ Compute and report metrics. Returns last metric: ndcg@100 """
    print(f"Evaluating with {Xval_out.shape[0]} users")
    scores = _compute_metrics(predictions, Xval_out)
    _report_metrics(scores)
    return scores[-1]


def _compute_metrics(predictions, Xval_out):
    """ Compute and return the three reported metrics as a list:
     recall@20, recall@100 and ndcg@100 (in that order). """
    recall_20_scores = recall_k(predictions, Xval_out, 20)
    avg_recall_20 = np.average(recall_20_scores)

    recall_100_scores = recall_k(predictions, Xval_out, 100)
    avg_recall_100 = np.average(recall_100_scores)

    ndcg_100_scores = ndcg_k(predictions, Xval_out, 100)
    avg_ndcg_100 = np.average(ndcg_100_scores)

    return [avg_recall_20, avg_recall_100, avg_ndcg_100]


def _report_metrics(metrics):
    """ Report the metrics in human friendly notation. """
    avg_recall_20, avg_recall_100, avg_ndcg_100 = metrics

    print(f"Average Recall@20", np.around(avg_recall_20, decimals=3))
    print(f"Average Recall@100", np.around(avg_recall_100, decimals=3))
    print(f"Average nDCG@100", np.around(avg_ndcg_100, decimals=3))


def simulate(alg, Xval_in, Xval_out, S, pos_feedback=2, strength=3, repeats=1):
    """ simulate each user multiple times (reduce variability due to tag sampling). """
    print(f"Evaluating with {Xval_out.shape[0]} users")
    all_scores = list()
    for repeat in range(repeats):
        scores = _simulate(alg, Xval_in, Xval_out, S, pos_feedback=pos_feedback, strength=strength)
        all_scores.append(scores)

    metrics = list(map(np.mean, zip(*all_scores)))
    _report_metrics(metrics)
    return metrics[-1]


def _simulate(alg, Xval_in, Xval_out, S, pos_feedback=2, strength=3):
    # all_predictions = list()
    tag_names = alg.tags[:-1]
    all_modifications = list()
    for u in range(Xval_in.shape[0]):
        # history = Xval_in[u].nonzero()[1]
        wanted = Xval_out[u].nonzero()[1]
        wanted_tags = np.asarray(S[wanted].sum(axis=0)).flatten()
        modifications = dict()

        if wanted_tags.sum() == 0:
            print("Warning: user with no wanted tags in test set.")
            all_modifications.append(modifications)
            continue

        wanted_tags = wanted_tags / wanted_tags.sum()
        for f in range(pos_feedback):
            tag = np.random.choice(tag_names, 1, p=wanted_tags)[0]
            modifications[tag] = strength
        # print(modifications)
        all_modifications.append(modifications)

        # predictions = alg.predict(history, modifications, retarget=False)
        # all_predictions.append(predictions)

    predictions = alg.predict_all_interactive(Xval_in, all_modifications, retarget=False)

    # predictions = np.stack(all_predictions, axis=0)
    return _compute_metrics(predictions, Xval_out)


def iterate_hyperparams(hyperparameter_ranges):
    combinations = [dict()]
    for param, values in hyperparameter_ranges.items():
        if not isinstance(values, Iterable):
            values = [values]
        new_combinations = list()
        for value in values:
            new_combinations.extend([{**combination, param: value} for combination in combinations])
        combinations = new_combinations
    return combinations


def gridsearch(Alg, Xtrain, S, Xval_in, Xval_out, hyperparameter_ranges, fit_params=dict()):
    best = (0, None)
    for hyperparameters in tqdm(iterate_hyperparams(hyperparameter_ranges)):
        tqdm.write(f"Training model {Alg.__name__} with hyperparameters {hyperparameters}")
        alg = Alg(**hyperparameters)
        alg.fit(Xtrain, S, **fit_params)
        # tqdm.write("Done, evaluating..")
        metric = eval(alg, Xval_in, Xval_out)
        payload = (metric, hyperparameters)
        best = max(best, payload, key=lambda x: x[0])
    return best


def get_rec_counts(algorithm, histories, k=100):
    """ Computes how many times each item was recommended in top k. """
    item_rec_counts = np.zeros(histories.shape[1])

    predictions = algorithm.predict_all(histories)
    predictions[histories.astype(bool).toarray()] = -10000
    recommendations = util.predictions_to_recommendations(predictions, top_k=k)
    for u in range(recommendations.shape[0]):
        item_rec_counts[recommendations[u]] += 1

    return item_rec_counts


def plot_rec_counts_l(*item_rec_counts_l, labels=[], k=100):
    """ Helper function to plot recommendation counts. See plot_long_tail. """
    n = 0
    for index, item_rec_counts in enumerate(item_rec_counts_l):
        n = max(n, item_rec_counts.shape[0])
        values = np.sort(item_rec_counts)[::-1]
        values[values == 0] = np.nan
        values = np.log(values)
        label = ""
        if index < len(labels):
            label = labels[index]
        plt.scatter(np.arange(values.shape[0]), values, s=1, label=label)

    plt.xlim(0, n)
    plt.xlabel('Item rank')
    plt.ylabel(f"log of recommendation count in top {k}")
    plt.legend()
    plt.show()


def plot_long_tail(*algs, histories, Xtest_out=None, labels=[], k=20):
    """ Plots the 'long tail' distribution of recommendation for multiple algorithms. """
    item_rec_counts_l = [get_rec_counts(alg, histories, k=k) for alg in algs]
    if Xtest_out is not None:
        test_counts = np.asarray(Xtest_out.sum(axis=0)).flatten()
        item_rec_counts_l.append(test_counts)

    plot_rec_counts_l(*item_rec_counts_l, labels=labels, k=k)
