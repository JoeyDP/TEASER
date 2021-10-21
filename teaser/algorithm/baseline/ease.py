import numpy as np
import scipy.sparse

from ..algorithm import Algorithm


class EASE(Algorithm):
    def __init__(self, l2: float = 200):
        super().__init__()
        self.l2 = l2

    def fit(self, X: scipy.sparse.csr_matrix, S: scipy.sparse.csr_matrix = None, tags = None):
        # Input checking
        X.eliminate_zeros()
        X = X.astype(np.int32)
        assert np.all(X.data == 1), "X should only contain binary values"

        m, n = X.shape

        # Compute P
        XTX = (X.T @ X).toarray()
        P = np.linalg.inv(XTX + self.l2 * np.identity(n))
        del XTX

        # Compute B
        B = np.identity(n) - P @ np.diag(1.0 / np.diag(P))
        B[np.diag_indices(B.shape[0])] = 0.0

        self.B_ = B

        return self

    def predict_all(self, X: scipy.sparse.csr_matrix, retarget: bool = False):
        """ Compute scores for a matrix of users (for offline evaluation) """
        # Input checking
        assert hasattr(self, "B_"), "fit needs to be called before predict"
        X.eliminate_zeros()
        assert np.all(X.data == 1), "X should only contain binary values"

        scores = X @ self.B_

        if not retarget:
            scores[X.toarray().astype(bool)] = -1e10

        return scores
