import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


def build_gaia_covariance(err_pmra, err_pmdec, err_plx,
                          corr_pmra_pmdec, corr_plx_pmra, corr_plx_pmdec):
    """Build an N x 3 x 3 observational covariance matrix."""
    N = len(err_pmra)
    Sigma_obs = np.zeros((N, 3, 3))
    Sigma_obs[:, 0, 0] = err_pmra ** 2
    Sigma_obs[:, 1, 1] = err_pmdec ** 2
    Sigma_obs[:, 2, 2] = err_plx ** 2

    cov_pmra_pmdec = err_pmra * err_pmdec * corr_pmra_pmdec
    cov_plx_pmra = err_plx * err_pmra * corr_plx_pmra
    cov_plx_pmdec = err_plx * err_pmdec * corr_plx_pmdec

    Sigma_obs[:, 0, 1] = Sigma_obs[:, 1, 0] = cov_pmra_pmdec
    Sigma_obs[:, 0, 2] = Sigma_obs[:, 2, 0] = cov_plx_pmra
    Sigma_obs[:, 1, 2] = Sigma_obs[:, 2, 1] = cov_plx_pmdec
    return Sigma_obs


def get_mahalanobis_prob(X, Sigma_obs, mu_c, Sigma_c, epsilon=1e-6):
    """Evaluate the cluster density including each star's measurement error."""
    N, D = X.shape
    probs = np.zeros(N)
    I = np.eye(D)
    for i in range(N):
        Sigma_total = Sigma_c + Sigma_obs[i] + epsilon * I
        try:
            probs[i] = multivariate_normal.pdf(
                X[i], mean=mu_c, cov=Sigma_total)
        except (np.linalg.LinAlgError, ValueError):
            probs[i] = 1e-12
    return probs


def heteroscedastic_gumm(X, Sigma_obs, max_iter=50, tol=1e-4):
    """Heteroscedastic Gaussian + uniform mixture solved with EM."""
    N, D = X.shape
    pi_c = 0.5
    mu_c = np.median(X, axis=0)
    Sigma_c = np.cov(X.T)
    if np.ndim(Sigma_c) == 0:
        Sigma_c = np.eye(D) * float(Sigma_c)

    spans = np.ptp(X, axis=0)
    spans = np.maximum(spans, 1e-8)
    volume = np.prod(spans)
    P_u = 1.0 / volume
    log_likelihoods = []

    gamma = np.full(N, 0.5)
    for iteration in range(max_iter):
        P_c = get_mahalanobis_prob(X, Sigma_obs, mu_c, Sigma_c)
        gamma = (pi_c * P_c) / (
            pi_c * P_c + (1 - pi_c) * P_u + 1e-12)

        N_c = np.sum(gamma)
        if N_c <= 1e-10:
            break
        pi_c = N_c / N
        mu_c_new = np.sum(gamma[:, np.newaxis] * X, axis=0) / N_c

        Sigma_c_new = np.zeros((D, D))
        for i in range(N):
            diff = (X[i] - mu_c_new).reshape(D, 1)
            Sigma_c_new += gamma[i] * (diff @ diff.T - Sigma_obs[i])
        Sigma_c_new /= N_c

        eigenvalues, eigenvectors = np.linalg.eigh(Sigma_c_new)
        eigenvalues[eigenvalues < 1e-6] = 1e-6
        Sigma_c = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        mu_c = mu_c_new

        ll = np.sum(np.log(pi_c * P_c + (1 - pi_c) * P_u + 1e-12))
        log_likelihoods.append(ll)
        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break

    return gamma, mu_c, Sigma_c


def GUMMProbs(clust_data, data_err=None, *args, **kwargs):
    """
    pyUPMASK bridge.

    When three per-star errors are available, use the heteroscedastic
    Gaussian+uniform model in (pmra, pmdec, parallax). Correlations are kept
    at zero in this experimental version because they are not present in the
    current params.ini input schema. If error data are unavailable or invalid,
    fall back to a two-component full-covariance GaussianMixture.
    """
    clust_data = np.asarray(clust_data, dtype=float)

    if data_err is not None:
        try:
            data_err = np.asarray(data_err, dtype=float)
            if (len(clust_data) == len(data_err) and clust_data.ndim == 2
                    and data_err.ndim == 2 and clust_data.shape[1] == 3
                    and data_err.shape[1] >= 3):
                err_pmra = data_err[:, 0]
                err_pmdec = data_err[:, 1]
                err_plx = data_err[:, 2]
                zeros = np.zeros_like(err_pmra)

                Sigma_obs = build_gaia_covariance(
                    err_pmra, err_pmdec, err_plx,
                    corr_pmra_pmdec=zeros,
                    corr_plx_pmra=zeros,
                    corr_plx_pmdec=zeros)
                gamma, _, _ = heteroscedastic_gumm(clust_data, Sigma_obs)
                return gamma
        except Exception:
            pass

    # Experimental fallback: two Gaussian components.  The denser component
    # (smaller covariance determinant) is treated as the cluster component.
    gmm = GaussianMixture(
        n_components=2, covariance_type='full', random_state=42)
    gmm.fit(clust_data)
    dets = [np.linalg.det(c) for c in gmm.covariances_]
    cluster_idx = int(np.argmin(dets))
    return gmm.predict_proba(clust_data)[:, cluster_idx]
