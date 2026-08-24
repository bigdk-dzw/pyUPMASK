"""Synthetic smoke test for the experimental pyUPMASK branch.

This is not an astrophysical validation. It only checks that HDBSCAN,
heteroscedastic GUMM, and adaptive spatial KDE execute together and separate a
simple injected compact population from a diffuse field.
"""
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from modules.GUMM import GUMMProbs
from modules.KDEAnalysis import probs as kde_probs


def main():
    rng = np.random.default_rng(7)
    n_cluster, n_field = 150, 450

    cluster = rng.multivariate_normal(
        [0.0, 0.0, 1.0], np.diag([0.10, 0.10, 0.04]) ** 2, n_cluster)
    field = np.column_stack([
        rng.uniform(-2, 2, n_field),
        rng.uniform(-2, 2, n_field),
        rng.uniform(0.2, 1.8, n_field),
    ])
    data = np.vstack([cluster, field])

    errors = np.column_stack([
        rng.uniform(0.02, 0.15, n_cluster + n_field),
        rng.uniform(0.02, 0.15, n_cluster + n_field),
        rng.uniform(0.01, 0.08, n_cluster + n_field),
    ])

    xy = np.vstack([
        rng.normal([0.5, 0.5], [0.07, 0.07], (n_cluster, 2)),
        rng.uniform(0, 1, (n_field, 2)),
    ])

    scaler = StandardScaler().fit(data)
    data_scaled = scaler.transform(data)
    errors_scaled = errors / scaler.scale_

    labels = HDBSCAN(min_cluster_size=25, min_samples=5).fit_predict(data_scaled)
    hdbscan_survive = labels != -1

    gumm_p = GUMMProbs(data_scaled, errors_scaled)
    upstream = np.where(hdbscan_survive, gumm_p, 0.0)
    final_p = kde_probs(xy, data, upstream)

    truth = np.r_[np.ones(n_cluster), np.zeros(n_field)]
    result = {
        "n_total": len(truth),
        "hdbscan_non_noise": int(hdbscan_survive.sum()),
        "final_nonzero": int((final_p > 0).sum()),
        "gumm_auc": float(roc_auc_score(truth, gumm_p)),
        "final_auc": float(roc_auc_score(truth, final_p)),
        "final_ap": float(average_precision_score(truth, final_p)),
        "final_brier": float(brier_score_loss(truth, final_p)),
        "cluster_mean_probability": float(final_p[:n_cluster].mean()),
        "field_mean_probability": float(final_p[n_cluster:].mean()),
    }

    for key, value in result.items():
        print(f"{key}: {value}")

    if result["final_auc"] < 0.95:
        raise SystemExit("Smoke test failed: final AUC < 0.95")
    if result["cluster_mean_probability"] <= result["field_mean_probability"]:
        raise SystemExit("Smoke test failed: cluster mean probability <= field mean")


if __name__ == "__main__":
    main()
