from typing import List

import numpy as np
import scipy.sparse
import scipy.linalg
import scipy.sparse.linalg

from tqdm.auto import tqdm

import teaser.util as util
from .algorithm import AspectAlgorithm


class WMF_S(AspectAlgorithm):
    """
    Model with matrix factorization iso EDLAE.
    Solved with numba and a trick based on the Woodbury matrix identity to learn the user factors faster.
    """
    def __init__(self, alpha: float = 5, l2: float = 0.05):
        super().__init__()
        self.alpha = alpha      # for confidence
        self.l2 = l2

    @property
    def amt_items(self):
        assert hasattr(self, "DT_"), "fit needs to be called before predict"
        return self.DT_.shape[1]

    @property
    def t(self):
        assert hasattr(self, "tags"), "fit needs to be called before predict"
        return len(self.tags)

    def fit(self, X: scipy.sparse.csr_matrix, S: scipy.sparse.csr_matrix=None, tags: List[str] = None):
        super().fit(X, S=S, tags=tags)

        # Input checking
        X.eliminate_zeros()
        assert np.all(X.data == 1), "X should only contain binary values"
        assert np.all(S.data == 1), "S should only contain binary values"

        X = X.astype(np.int32)
        X = X.copy()

        m, n = X.shape
        t = S.shape[1]

        # Center
        self.centers = np.asarray(X.sum(axis=0) / m).flatten()
        self.pop_scale = np.max(self.centers)
        self.centers = self.centers / self.pop_scale

        DT = S.T.astype(np.int32)
        self.DT_ = DT
        self.DT_ = scipy.sparse.vstack((self.DT_, self.centers)).tocsr()
        t += 1

        DTD = (self.DT_ @ self.DT_.T).toarray()
        self.DTD_ = DTD
        self.DTDinv_ = np.linalg.inv(DTD + self.l2 * np.identity(t))
        self.DT_ = self.DT_.toarray()

        return self

    def _user_vector_x(self, x):
        x = x.astype(np.float64)

        Cu_diag = 1 + x * self.alpha

        DT = self.DT_
        indices = np.where(x > 0)[0]
        Dsub = DT[:, indices]

        # Woodbury matrix identity
        GDT = self.DTDinv_ @ Dsub
        Pu = np.linalg.inv(Dsub.T @ self.DTDinv_ @ Dsub + (1 / self.alpha) * np.identity(len(indices)))
        Pu = self.DTDinv_ - GDT @ Pu @ GDT.T

        Pu = Pu @ (DT * Cu_diag @ x[:, np.newaxis])
        Pu = Pu.flatten()

        return Pu

    def _user_vector(self, history):
        x = np.zeros(self.amt_items)
        x[history] = 1
        return self._user_vector_x(x)

    # for caching of test set user vectors (speed up simulated experiments)
    _cached_X: scipy.sparse.csr_matrix
    _cached_factors: np.ndarray

    def clear_cache(self):
        del self._cached_X
        del self._cached_factors

    def _user_vectors(self, X):
        if hasattr(self, '_cached_X') and X is self._cached_X:
            return self._cached_factors.copy()

        # start = datetime.now()
        factors = user_vectors(X.data, X.indptr, X.indices, self.DT_, self.alpha, self.DTDinv_)
        # print("time", datetime.now() - start)

        # self.DT_ = scipy.sparse.csr_matrix(self.DT_)
        # factors = np.vstack([self._user_vector_x(X[u].toarray().flatten()) for u in tqdm(range(X.shape[0]))])

        # assert np.allclose(factors, factors2)

        self._cached_X = X
        self._cached_factors = factors.copy()

        return factors

    def predict_all(self, X: scipy.sparse.csr_matrix, retarget: bool = False):
        """ Compute scores for a matrix of users (for offline evaluation) """
        # Input checking
        assert hasattr(self, "DT_"), "fit needs to be called before predict"
        X.eliminate_zeros()
        assert np.all(X.data == 1), "X should only contain binary values"

        P = self._user_vectors(X)
        scores = P @ self.DT_

        if not retarget:
            scores[X.toarray().astype(bool)] = -1e10

        return scores


from numba import prange, njit


@njit(parallel=True, fastmath=True)
def user_vectors(Xval, Xindptr, Xindices, DT, alpha, DTDinv):
    m = len(Xindptr) - 1
    t, n = DT.shape
    factors = np.zeros((m, t), dtype=np.float64)

    for row in prange(m):
        # print("row", row)
        # x = np.zeros(n, dtype=np.float64)
        start, end = Xindptr[row], Xindptr[row+1]
        indices = Xindices[start:end]
        # x[indices] = 1
        factors[row] = user_vector(indices, DT, alpha, DTDinv)

    return factors


@njit(parallel=True, fastmath=True)
def user_vector(indices, DT, alpha, DTDinv):
    # Cu_diag = 1 + x * alpha

    # indices = np.where(x > 0)[0]
    Dsub = DT[:, indices]

    # Woodbury matrix identity
    GDT = DTDinv @ Dsub
    Pu = np.linalg.inv(Dsub.T @ DTDinv @ Dsub + (1 / alpha) * np.identity(len(indices)))
    Pu = DTDinv - GDT @ Pu @ GDT.T

    Cu_diag = (1 + alpha) * np.ones(len(indices))

    # Pu = Pu @ (DT @ (Cu_diag * x))
    Pu = Pu @ (Dsub @ Cu_diag)

    Pu = Pu.flatten()

    return Pu
