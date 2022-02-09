from typing import List

import numpy as np
import scipy.sparse

from .teaser_s import TEASER_S
from .baseline.ease import EASE


class TEASER_EASE(TEASER_S):
    """
    Combination of TEASER and EASE by taking product of positive scores.
    """
    def __init__(self, l2_ease: float = 200,
                 l2_1: float = 0.05, l2_2: float = 0.05, rho: float = 0.05, delta: float = 0,
                 max_iterations: int = 10
         ):
        super().__init__(l2_1=l2_1, l2_2=l2_2, rho=rho, delta=delta, max_iterations=max_iterations)
        self.ease = EASE(l2=l2_ease)

    def fit(self, X: scipy.sparse.csr_matrix, S: scipy.sparse.csr_matrix = None, tags: List[str] = None,
            XTX=None, p=None, U=None      # can be preprocessed
            ):
        super().fit(X, S, tags, XTX=XTX, p=p, U=U)
        print("Fitting EASE")
        self.ease.fit(X)
        return self

    # For simplicity, only predict_all is correctly implemented. If used for explaining, some other functions in
    # TEASER-S should be overriden
    def predict_all(self, X: scipy.sparse.csr_matrix, retarget: bool = False):
        """ Compute scores for a matrix of users (for offline evaluation) """
        # Input checking
        assert hasattr(self, "E_"), "fit needs to be called before predict"
        X.eliminate_zeros()
        assert np.all(X.data == 1), "X should only contain binary values"

        scores1 = super().predict_all(X, retarget=True)
        scores1[scores1 < 0] = 0        # make sure there are no negative scores (for geometric mean)
        scores2 = self.ease.predict_all(X, retarget=True)
        scores2[scores2 < 0] = 0        # make sure there are no negative scores (for geometric mean)
        scores = np.multiply(scores1, scores2)

        if not retarget:
            scores[X.toarray().astype(bool)] = -1e10

        return scores

    def predict_all_interactive(self, X: scipy.sparse.csr_matrix, modifications=list(), retarget: bool = False):
        scores1 = super().predict_all_interactive(X, modifications, retarget=True)
        scores1[scores1 < 0] = 0        # make sure there are no negative scores (for geometric mean)
        scores2 = self.ease.predict_all(X, retarget=True)
        scores2[scores2 < 0] = 0        # make sure there are no negative scores (for geometric mean)
        scores = np.multiply(scores1, scores2)

        if not retarget:
            scores[X.toarray().astype(bool)] = -1e10

        return scores