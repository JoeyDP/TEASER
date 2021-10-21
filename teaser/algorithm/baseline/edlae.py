from typing import List

import numpy as np
import scipy.sparse
import scipy.stats

from tqdm.auto import tqdm

import teaser.util as util
from ..algorithm import Algorithm


class EDLAE(Algorithm):
    def __init__(self, l2: float = 0.05, rho: float = 0.05, init_scale=0.05, t=3,
                 max_iterations: int = 10, rho_fact: float = 2, res_margin: float = 10):
        """
        :param l2: regularization
        :param rho: step size of ADMM
        :param max_iterations: Maximum amount of iterations before stopping the algorithm
        :param init_scale: Scale of random initialization of E and D
        """
        super().__init__()
        self.l2 = l2
        self.rho = rho
        self.init_scale = init_scale
        self.t = t
        self.max_iterations = max_iterations
        self.rho_fact = rho_fact
        self.res_margin = res_margin

    @property
    def amt_items(self):
        assert hasattr(self, "DT_"), "fit needs to be called first"
        return self.DT_.shape[1]

    @staticmethod
    def loss(l2, rho, X, E, DT, beta, gamma):
        """ Loss value (without delta) """
        EDT = E @ DT
        l = np.linalg.norm(X - X @ (EDT - np.diag(beta))) ** 2
        l += l2 * np.linalg.norm(EDT - np.diag(beta)) ** 2
        l += rho * np.linalg.norm(beta - np.diag(EDT) + gamma) ** 2
        return l

    def fit(self, X: scipy.sparse.csr_matrix, S: scipy.sparse.csr_matrix = None, tags = None,
            XTX=None
            ):
        # Input checking
        X.eliminate_zeros()
        X = X.astype(np.int32)
        assert np.all(X.data == 1), "X should only contain binary values"

        m, n = X.shape

        # Intitialization
        beta = np.zeros(n)
        gamma = np.zeros(n)

        DT = np.random.randn(self.t, n) * self.init_scale

        diag_indices_n = np.diag_indices(n)

        if XTX is None:
            XTX = X.T.dot(X).toarray()#.astype(np.float64)

        XTX_diag = np.diag(XTX).copy()
        P = np.linalg.inv(XTX + self.l2 * np.identity(n))

        # ADMM iterations
        loop = tqdm(range(self.max_iterations))
        for it in loop:
            # Compute common
            F = XTX * (1 + beta)
            F[diag_indices_n] += self.rho * (beta + gamma) + self.l2 * beta

            # Compute E
            E = P @ (F @ (DT.T @ np.linalg.inv(DT @ DT.T)))

            DT = E.T @ (XTX + self.l2 * np.identity(n)) @ E
            DT = np.linalg.inv(DT)
            DT = DT @ np.asarray(E.T @ F)

            # Compute beta
            EDT_diag = util.diag_dot(E, DT)

            # OLD: separate reg
            # beta = (util.diag_dot(XTX.T @ E, DT) - XTX_diag + self.rho * EDT_diag - self.rho * gamma)
            # beta /= (XTX_diag - self.l2_1 + 2 * self.rho)

            # combined reg
            beta = (util.diag_dot(XTX.T @ E, DT) - XTX_diag + (self.rho + self.l2) * EDT_diag - self.rho * gamma)
            beta /= (XTX_diag + self.l2 + 2 * self.rho)

            # We prefer a positive diagonal
            beta[beta < 0] = 0

            # Compute gamma
            gamma += beta - EDT_diag

            loop.write(f"norm E {np.linalg.norm(E)}")
            loop.write(f"norm D {np.linalg.norm(DT)}")
            # l = EDLAE.loss(self.l2, self.rho, X, E, DT, beta, gamma)
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

