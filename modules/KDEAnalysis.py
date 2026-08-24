import numpy as np
from scipy.stats import gaussian_kde


def adaptive_kde(X, X_eval=None, alpha=0.5, base_h0=None):
    """Adaptive KDE using a pilot estimate and local bandwidth factors."""
    X = np.asarray(X, dtype=float)
    if X_eval is None:
        X_eval = X
    else:
        X_eval = np.asarray(X_eval, dtype=float)

    N, D = X.shape
    M = X_eval.shape[0]

    if np.all(X == X[0]):
        return np.ones(M)

    pilot_kde = gaussian_kde(X.T, bw_method=base_h0)
    pilot_density = pilot_kde.evaluate(X.T)
    pilot_density = np.clip(pilot_density, a_min=1e-12, a_max=None)

    g = np.exp(np.mean(np.log(pilot_density)))
    lambda_i = (pilot_density / g) ** (-alpha)

    if base_h0 is None:
        h0 = pilot_kde.factor * np.mean(np.std(X, axis=0))
    else:
        h0 = float(base_h0)

    if not np.isfinite(h0) or h0 <= 0:
        h0 = 1e-4

    h_i = np.maximum(h0 * lambda_i, 1e-8)
    densities = np.zeros(M)
    norm_const = (2 * np.pi) ** (D / 2.0)

    for i in range(M):
        diff = X_eval[i] - X
        dist_sq = np.sum(diff ** 2, axis=1)
        kernel_vals = np.exp(-dist_sq / (2 * h_i ** 2)) / (
            (h_i ** D) * norm_const)
        densities[i] = np.mean(kernel_vals)

    return densities


def probs(xy, data, cl_probs):
    """
    Experimental pyUPMASK bridge.

    Adaptive KDEs are evaluated only in normalized (x,y). The submitted
    experimental fusion rule P_final = P_upstream * P_spatial is preserved.
    """
    xy = np.asarray(xy, dtype=float)
    cl_probs = np.asarray(cl_probs, dtype=float)

    if cl_probs.size == 0:
        return cl_probs

    # The submitted median split is retained, but the median is computed on
    # positive upstream probabilities. Otherwise, when field stars dominate,
    # median(cl_probs)==0 would incorrectly put every zero-probability field
    # star into the member KDE and disable the experiment entirely.
    positive = cl_probs[cl_probs > 0]
    if positive.size < 3:
        return cl_probs
    threshold = np.median(positive)
    memb_mask = cl_probs >= threshold
    non_memb_mask = cl_probs < threshold

    if np.sum(memb_mask) < 3 or np.sum(non_memb_mask) < 3:
        return cl_probs

    xy_memb = xy[memb_mask]
    xy_non = xy[non_memb_mask]

    try:
        dens_memb = adaptive_kde(xy_memb, X_eval=xy)
        dens_non = adaptive_kde(xy_non, X_eval=xy)
    except (np.linalg.LinAlgError, ValueError):
        return cl_probs

    total_dens = dens_memb + dens_non + 1e-12
    kde_prob_spatial = dens_memb / total_dens

    final_probs = cl_probs * kde_prob_spatial
    return np.clip(final_probs, 0.0, 1.0)
