import numpy as np
import scipy.sparse

from ..algorithm import Algorithm

from implicit.als import AlternatingLeastSquares


class WMF(Algorithm):
    def __init__(self, l2: float = 200, t=200, alpha=10, max_iterations=10):
        super().__init__()
        self.l2 = l2
        self.t = t
        self.alpha = alpha
        self.max_iterations = max_iterations

    def fit(self, X: scipy.sparse.csr_matrix, S: scipy.sparse.csr_matrix = None, tags = None):
        # Input checking
        X.eliminate_zeros()
        X = X.astype(np.int32)
        assert np.all(X.data == 1), "X should only contain binary values"

        # initialize a model
        model = AlternatingLeastSquares(factors=self.t, regularization=self.l2, iterations=self.max_iterations)

        C = X.T * self.alpha

        # train the model on a sparse matrix of item/user/confidence weights
        model.fit(C)

        self.model = model

        return self

    def predict_all(self, X: scipy.sparse.csr_matrix, retarget: bool = False):
        """ Compute scores for a matrix of users (for offline evaluation) """
        # Input checking
        X.eliminate_zeros()
        assert np.all(X.data == 1), "X should only contain binary values"

        m, n = X.shape

        C = X * self.alpha

        # item_factors = self.model.item_factors
        # user_factors = np.zeros((m, self.t))
        # for u in range(m):
        #     user_factors[u] = self.model.recalculate_user(u, C)
        #
        # scores = user_factors @ item_factors.T

        recommendations = self.model.recommend_all(C, N=n, recalculate_user=True, filter_already_liked_items=False)
        # print(recommendations)

        template = np.linspace(0, 1, n)[::-1]
        scores = np.zeros(X.shape)
        for u in range(m):
            scores[u, recommendations[u]] = template

        if not retarget:
            scores[X.toarray().astype(bool)] = -1e10

        return scores
