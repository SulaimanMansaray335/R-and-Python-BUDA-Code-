# rpowertransform/multivariate.py
import numpy as np
from numpy.linalg import lstsq, slogdet
from dataclasses import dataclass

from .transforms import (
    box_cox_transform,
    box_cox_log_jacobian,
    yeo_johnson_transform,
    eo_johnson_log_jacobian,
)

try:
    from scipy.optimize import minimize
except ImportError as e:
    raise ImportError(
        "rpowertransform requires SciPy. Install with `pip install scipy`."
    ) from e


@dataclass
class MultivariatePowerTransform:
    """
    Multivariate Box–Cox / Yeo–Johnson power transform with ML estimation
    of λ, modeled after R's car::powerTransform.

    Maximizes profile log-likelihood:

        ℓ(λ) = - (n/2) * log|Σ̂(λ)| + log J(λ)

    where Σ̂(λ) is the residual covariance matrix from multivariate
    regression of transformed Y on X, and log J is the Jacobian term
    from the Box–Cox or Yeo–Johnson family.

    Parameters
    ----------
    family : {'box-cox', 'yeo-johnson'}
        Transformation family. 'box-cox' assumes Y > 0.
    lam_bounds : tuple(float, float)
        Bounds for each λ_j. Default (-3, 3) matches car defaults.
    ridge : float
        Small ridge added to Σ̂ for numerical stability.
    maxiter : int
        Max iterations for optimizer.
    tol : float
        Tolerance for optimizer.
    optimizer : str
        SciPy optimizer ('L-BFGS-B' is a good default).
    """

    family: str = "box-cox"
    lam_bounds: tuple = (-3.0, 3.0)
    ridge: float = 1e-8
    maxiter: int = 200
    tol: float = 1e-6
    optimizer: str = "L-BFGS-B"

    # Attributes set after fit()
    lambdas_: np.ndarray | None = None
    log_likelihood_: float | None = None
    coef_: np.ndarray | None = None # B̂ (k x p)
    sigma_: np.ndarray | None = None # Σ̂ (p x p)
    n_samples_: int | None = None
    n_targets_: int | None = None
    n_predictors_: int | None = None

    def _check_family(self):
        fam = self.family.lower()
        if fam not in ("box-cox", "yeo-johnson"):
            raise ValueError(
                "family must be 'box-cox' or 'yeo-johnson', "
                f"got {self.family!r}"
            )
        return fam

    def _prepare_xy(self, Y, X):
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]

        n, p = Y.shape

        if X is None:
            X = np.ones((n, 1), dtype=float)
        else:
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X[:, None]
            if X.shape[0] != n:
                raise ValueError("X and Y must have the same number of rows.")

        return Y, X, n, p, X.shape[1]

    # ---- Core loglik objective ----

    def _transform_and_log_jacobian_vec(self, Y, lam_vec, family):
        """
        Apply column-wise transform to Y and sum log-Jacobian contributions.

        Parameters
        ----------
        Y : (n, p) array
        lam_vec : (p,) array
        family : 'box-cox' or 'yeo-johnson'

        Returns
        -------
        Z : (n, p) transformed data
        logJ : float, total log-Jacobian
        """
        Y = np.asarray(Y, dtype=float)
        lam_vec = np.asarray(lam_vec, dtype=float)

        n, p = Y.shape
        if lam_vec.shape[0] != p:
            raise ValueError(
                f"lam_vec must be length {p}, got {lam_vec.shape[0]}"
            )

        Z = np.empty_like(Y, dtype=float)
        logJ = 0.0

        for j in range(p):
            yj = Y[:, j]
            lam = lam_vec[j]

            if family == "box-cox":
                zj = box_cox_transform(yj, lam)
                logJ += box_cox_log_jacobian(yj, lam)
            else: # yeo-johnson
                zj = yeo_johnson_transform(yj, lam)
                logJ += yeo_johnson_log_jacobian(yj, lam)

            Z[:, j] = zj

        return Z, logJ

    def _loglik(self, lam_vec, Y, X, family):
        """
        Profile log-likelihood for given λ vector.

        ℓ(λ) = - (n/2) log |Σ̂(λ)| + log J(λ)
        """
        lam_vec = np.asarray(lam_vec, dtype=float)

        # Transform and Jacobian
        Z, logJ = self._transform_and_log_jacobian_vec(Y, lam_vec, family)

        n, p = Z.shape

        # Multivariate regression: Z = X B + E
        # B̂ = (X'X)^(-1) X'Z (via least squares)
        B_hat, *_ = lstsq(X, Z, rcond=None)
        E = Z - X @ B_hat
        Sigma_hat = (E.T @ E) / n

        # Ridge for numerical stability
        Sigma_hat = Sigma_hat + self.ridge * np.eye(p)

        sign, logdet = slogdet(Sigma_hat)
        if sign <= 0:
            # Bad / singular covariance, heavily penalize
            return -np.inf, Sigma_hat, B_hat

        loglik = -0.5 * n * logdet + logJ
        return loglik, Sigma_hat, B_hat

    def _neg_loglik_for_opt(self, lam_vec, Y, X, family):
        loglik, _, _ = self._loglik(lam_vec, Y, X, family)
        # we minimize negative log-likelihood
        return -loglik

# ---- Public API ----

    def fit(self, Y, X=None):
        """
        Fit multivariate power transform λ via ML, like car::powerTransform.

        Parameters
        ----------
        Y : array-like, shape (n_samples, n_targets) or (n_samples,)
            Response(s) to transform.
        X : array-like, shape (n_samples, n_predictors), optional
            Design matrix on the right-hand side. If None, uses only an
            intercept (i.e., transforms unconditional Y).

        Returns
        -------
        self
        """
        family = self._check_family()
        Y, X, n, p, k = self._prepare_xy(Y, X)

        # Initial λ: 1 for Box-Cox, 0 for Yeo–Johnson (roughly matches R usage)
        if family == "box-cox":
            lam0 = np.ones(p, dtype=float)
        else:
            lam0 = np.zeros(p, dtype=float)

        bounds = [self.lam_bounds] * p

        res = minimize(
            self._neg_loglik_for_opt,
            x0=lam0,
            args=(Y, X, family),
            method=self.optimizer,
            bounds=bounds,
            options={"maxiter": self.maxiter, "ftol": self.tol},
        )

        if not res.success:
            raise RuntimeError(
                f"Optimization failed: {res.message}; "
                f"last λ = {res.x}"
            )

        lam_hat = res.x
        loglik_hat, Sigma_hat, B_hat = self._loglik(lam_hat, Y, X, family)

        self.lambdas_ = lam_hat
        self.log_likelihood_ = loglik_hat
        self.coef_ = B_hat
        self.sigma_ = Sigma_hat
        self.n_samples_ = n
        self.n_targets_ = p
        self.n_predictors_ = k

        return self

    def transform(self, Y):
        """
        Apply the fitted power transform to new Y.

        Parameters
        ----------
        Y : array-like, shape (n_samples, n_targets) or (n_samples,)
            Data to transform. Columns must correspond to the same variables
            used in `fit`.

        Returns
        -------
        Z : ndarray of shape (n_samples, n_targets)
        """
        if self.lambdas_ is None:
            raise RuntimeError("Call fit() before transform().")

        family = self._check_family()
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]

        if Y.shape[1] != self.lambdas_.shape[0]:
            raise ValueError(
                f"Expected {self.lambdas_.shape[0]} columns, got {Y.shape[1]}"
            )

        Z, _ = self._transform_and_log_jacobian_vec(Y, self.lambdas_, family)
        return Z

    def fit_transform(self, Y, X=None):
        """
        Convenience: fit and then transform Y.
        """
        self.fit(Y, X=X)
        return self.transform(Y)