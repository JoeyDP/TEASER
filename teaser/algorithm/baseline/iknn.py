import numpy as np
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity

from ..algorithm import Algorithm


class ItemKNN(Algorithm):
    def __init__(self, k: int = 200, normalize=False):
        super().__init__()
        self.k = k
        self.normalize = normalize

    def fit(self, X: scipy.sparse.csr_matrix, S: scipy.sparse.csr_matrix = None, tags = None):
        # Input checking
        X.eliminate_zeros()
        assert np.all(X.data == 1), "X should only contain binary values"

        m, n = X.shape

        item_cosine_similarities_ = cosine_similarity(X.T, dense_output=True)

        # Set diagonal to 0, because we don't want to support self similarity
        np.fill_diagonal(item_cosine_similarities_, 0)

        if self.k:
            top_k_per_row = np.argpartition(item_cosine_similarities_, -self.k, axis=1)[:, -self.k:]
            values = np.take_along_axis(item_cosine_similarities_, top_k_per_row, axis=1)

            res = scipy.sparse.lil_matrix(item_cosine_similarities_.shape)
            np.put_along_axis(res, top_k_per_row, values, axis=1)
            item_cosine_similarities_ = res.tocsr()

        if self.normalize:
            # normalize per row
            row_sums = item_cosine_similarities_.sum(axis=1)
            item_cosine_similarities_ = item_cosine_similarities_ / row_sums

        self.B_ = scipy.sparse.csr_matrix(item_cosine_similarities_)

        return self

    def predict_all(self, X: scipy.sparse.csr_matrix, retarget: bool = False):
        """ Compute scores for a matrix of users (for offline evaluation) """
        # Input checking
        assert hasattr(self, "B_"), "fit needs to be called before predict"
        X.eliminate_zeros()
        assert np.all(X.data == 1), "X should only contain binary values"

        scores = (X @ self.B_).toarray()

        if not retarget:
            scores[X.toarray().astype(bool)] = -1e10

        return scores
