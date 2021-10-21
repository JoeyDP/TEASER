from pathlib import Path
from typing import List

import numpy as np
import scipy.sparse
import scipy.linalg
import scipy.stats

from tqdm.auto import tqdm

import teaser.util as util
from .algorithm import AspectAlgorithm


class TEASER_S(AspectAlgorithm):
    """
    Model with ||X - X(ED^T - dm(beta))|| + l2 _1||ED^T - dm(beta)|| + l2_2 ||E||
    with D^T = vstack(D'^T, centers) (adds popularity feature)
    s.t. beta = diag(ED^T) and beta > 0
    """
    def __init__(self, l2_1: float = 0.05, l2_2: float = 0.05, rho: float = 0.05, delta: float = 0,
                 max_iterations: int = 10, rho_fact: float = 2, res_margin: float = 10):
        """
        :param l2_1: regularization
        :param rho: step size of ADMM
        :param delta: importance of similarity learned from side information
        :param l2_2: regularization on E
        :param max_iterations: Maximum amount of iterations before stopping the algorithm
        """
        super().__init__()
        self.l2_1 = l2_1
        self.l2_2 = l2_2
        self.rho = rho
        self.delta = delta
        self.max_iterations = max_iterations
        self.rho_fact = rho_fact
        self.res_margin = res_margin

    @property
    def amt_items(self):
        assert hasattr(self, "DT_"), "fit needs to be called first"
        return self.DT_.shape[1]

    @staticmethod
    def loss(l2_1, l2_2, rho, X, S, E, DT, beta, gamma):
        """ Loss value (without delta) """
        EDT = E @ DT
        l = np.linalg.norm(X - X @ (EDT - np.diag(beta))) ** 2
        l += l2_1 * np.linalg.norm(EDT - np.diag(beta)) ** 2
        l += l2_2 * (np.linalg.norm(E) ** 2 + scipy.sparse.linalg.norm(DT) ** 2)
        l += rho * np.linalg.norm(beta - np.diag(EDT) + gamma) ** 2
        return l

    def fit(self, X: scipy.sparse.csr_matrix, S: scipy.sparse.csr_matrix = None, tags: List[str] = None,
            XTX=None, p=None, U=None      # can be preprocessed
            ):
        super().fit(X, S=S, tags=tags)
        # Input checking
        X.eliminate_zeros()
        X = X.astype(np.int32)
        assert np.all(X.data == 1), "X should only contain binary values"
        assert np.all(S.data == 1), "S should only contain binary values"

        # X = X.copy()

        m, n = X.shape
        t = S.shape[1]

        # Preprocessing
        centers = np.asarray(X.sum(axis=0) / m).flatten()
        pop_scale = np.max(centers)
        centers = centers / pop_scale

        if self.delta != 0:
            # X = scipy.sparse.vstack((X, np.sqrt(self.delta / t) * S.T))
            X = scipy.sparse.vstack((X, np.sqrt(self.delta) * S.T))
            m = m + t

        # Intitialization
        beta = np.zeros(n)
        gamma = np.zeros(n)

        DT = S.T.copy().astype(np.float64)
        DT = scipy.sparse.vstack((DT, centers)).tocsr()

        # Precompute
        DTD = (DT @ DT.T).toarray()

        diag_indices_n = np.diag_indices(n)

        if XTX is None:
            XTX = X.T.dot(X).toarray()#.astype(np.float64)

        XTX_diag = np.diag(XTX).copy()

        if p is None or U is None:
            print("Decompose XTX")
            p, U = np.linalg.eigh(XTX)

        # add regularization to decomposition (U diag(p) U^T + U l2 * I U^T)
        p = p + self.l2_1

        print("Decompose DTD")
        q, V = np.linalg.eigh(DTD)

        G = (1 / (np.outer(p, q) + self.l2_2))

        # ADMM iterations
        loop = tqdm(range(self.max_iterations))
        for it in loop:
            # Compute E
            F = XTX * (1 + beta)
            F[diag_indices_n] += self.rho * (beta + gamma) + self.l2_1 * beta
            F = F @ DT.T

            F = U.T @ F @ V
            Y = F * G
            del F

            E = U @ Y @ V.T
            del Y

            # Compute beta
            EDT_diag = util.diag_dot(E, DT)

            # OLD: separate reg
            # beta = (util.diag_dot(XTX.T @ E, DT) - XTX_diag + self.rho * EDT_diag - self.rho * gamma)
            # beta /= (XTX_diag - self.l2_1 + 2 * self.rho)

            # combined reg
            beta = (util.diag_dot(XTX.T @ E, DT) - XTX_diag + (self.rho + self.l2_1) * EDT_diag - self.rho * gamma)
            beta /= (XTX_diag + self.l2_1 + 2 * self.rho)

            # We prefer a positive diagonal
            beta[beta < 0] = 0

            # Compute gamma
            gamma += beta - EDT_diag

            loop.write(f"norm E {np.linalg.norm(E)}")
            # loop.write(f"norm D {scipy.sparse.linalg.norm(DT)}")
            # l = TEASER_S.loss(self.l2_1, self.l2_2, self.rho, X, S, E, DT, beta, gamma)
            # loop.write(f"loss {l}")
            # loop.write(f"gamma {gamma}")

            diag_diff = np.linalg.norm(beta - EDT_diag)
            loop.write(f"diag norm: {np.linalg.norm(EDT_diag)}")
            loop.write(f"diag_diff: {diag_diff}")

            if hasattr(self, 'E_'):
                primal_res = diag_diff
                dual_res = np.linalg.norm(self.rho * (self.E_ - E))
                loop.write(f"rho * change E: {dual_res}")

                # if primal_res > self.res_margin * dual_res:
                #     self.rho *= self.rho_fact
                #     loop.write(f"rho changed {self.rho}")
                #     gamma /= self.rho_fact
                # elif dual_res > self.res_margin * primal_res:
                #     self.rho /= self.rho_fact
                #     loop.write(f"rho changed {self.rho}")
                #     gamma *= self.rho_fact

            loop.write("")
            self.E_ = E

        self.DT_ = DT

        # based on weighted selection of one item in history
        # Can be used to estimate profile probability a sampling from E by multiplying by history length
        # assumes sampling with replacement
        self.factor_average = np.average(self.E_, weights=centers, axis=0)

        return self

    def predict_all(self, X: scipy.sparse.csr_matrix, retarget: bool = False):
        """ Compute scores for a matrix of users (for offline evaluation) """
        # Input checking
        assert hasattr(self, "E_"), "fit needs to be called before predict"
        X.eliminate_zeros()
        assert np.all(X.data == 1), "X should only contain binary values"

        scores = X @ self.E_ @ self.DT_

        if not retarget:
            scores[X.toarray().astype(bool)] = -1e10

        return scores

    def _user_vector(self, history):
        user_vec = np.asarray(self.E_[history].sum(axis=0)).flatten()
        return user_vec

    def certainty(self, history, user_vec):
        """ Different measure of certainty: how much would user vector change if the average item factor is added. """
        abs_uv = np.abs(user_vec)
        certainty = np.average(abs_uv / (abs_uv + np.abs(self.factor_average)))
        return certainty

    def save(self, path: Path):
        data = {
            'kwargs': {
                'l2_1': self.l2_1,
                'l2_2': self.l2_2,
                'rho': self.rho,
                'delta': self.delta,
            },
            'weights': {
                'E': self.E_,
                'DT': self.DT_,
                'avg': self.factor_average,
            },
            'info': {
                'tags': self.tags,
                'features': self.features,
                'features_index': self.features_index,
            }
        }
        np.savez(path, **data)

    @classmethod
    def load(cls, path: Path):
        # numpy auto adds this extension
        if path.suffix != "npz":
            path = path.with_suffix('.npz')

        with np.load(path, allow_pickle=True) as data:
            kwargs = data['kwargs'][()]         # convert kwargs to dict
            alg = TEASER_S(**kwargs)

            weights = data['weights'][()]       # convert weights to dict
            alg.E_ = weights['E']
            alg.DT_ = weights['DT']
            # alg.factor_std = weights['std']
            alg.factor_average = weights['avg']

            info = data['info'][()]             # convert info to dict
            alg.tags = info['tags']
            alg.features = info['features']
            alg.features_index = info['features_index']
            return alg

