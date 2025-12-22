# rpowertransform/transforms.py
import numpy as np


EPS = 1e-8


def box_cox_transform(u, lam):
    """
    Box-Cox scaled power transform:
    (u^lam - 1) / lam if lam != 0
    log(u) if lam == 0

    Assumes u > 0.
    """
    u = np.asarray(u, dtype=float)
    if np.any(u <= 0):
    raise ValueError("Box-Cox requires strictly positive data (u > 0).")

    if abs(lam) < EPS:
    z = np.log(u)
    else:
    z = (np.power(u, lam) - 1.0) / lam
    return z


def box_cox_log_jacobian(u, lam):
    """
    log |Jacobian| term for Box-Cox over vector u:

    log J = (lam - 1) * sum(log(u))
    """
    u = np.asarray(u, dtype=float)
    if np.any(u <= 0):
    raise ValueError("Box-Cox requires strictly positive data (u > 0).")
    return (lam - 1.0) * np.sum(np.log(u))


def yeo_johnson_transform(u, lam):
    """
    Yeo–Johnson transform (car::yjPower style):

    For y >= 0:
    if lam != 0: ((y + 1)^lam - 1) / lam
    if lam == 0: log(y + 1)

    For y < 0:
    if lam != 2: -((1 - y)^(2 - lam) - 1) / (2 - lam)
    if lam == 2: -log(1 - y)

    Works for any real u.
    """
    u = np.asarray(u, dtype=float)
    z = np.empty_like(u, dtype=float)

    pos = u >= 0
    neg = ~pos

    # y >= 0 part
    if abs(lam) < EPS:
    z[pos] = np.log1p(u[pos]) # log(1 + y)
    else:
    z[pos] = (np.power(u[pos] + 1.0, lam) - 1.0) / lam

    # y < 0 part
    if abs(lam - 2.0) < EPS:
    z[neg] = -np.log1p(-u[neg]) # -log(1 - y)
    else:
    z[neg] = -(
    (np.power(1.0 - u[neg], 2.0 - lam) - 1.0)
    / (2.0 - lam)
    )

    return z


def yeo_johnson_log_jacobian(u, lam):
    """
    log |Jacobian| for Yeo–Johnson over vector u.

    From derivative:

    For y >= 0: dz/dy = (1 + y)^(lam - 1)
    For y < 0: dz/dy = (1 - y)^(1 - lam)

    So:

    log J = sum_{y>=0} (lam - 1) * log(1 + y)
    + sum_{y<0} (1 - lam) * log(1 - y)
    """
    u = np.asarray(u, dtype=float)
    pos = u >= 0
    neg = ~pos

    logJ = 0.0
    if np.any(pos):
    logJ += (lam - 1.0) * np.sum(np.log1p(u[pos]))
    if np.any(neg):
    logJ += (1.0 - lam) * np.sum(np.log1p(-u[neg]))
    return logJ