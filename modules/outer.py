import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from . import inner
from .GUMM import GUMMProbs
from .GUMMExtras import GUMMProbCut
from . import KDEAnalysis


def loop(
    ID, xy, data, data_err, resampleFlag, PCAflag, PCAdims, GUMM_flag,
    GUMM_perc, KDEP_flag, IL_runs, N_membs, N_cl_max, clust_method,
    clRjctMethod, Kest, C_thresh, cl_method_pars, prfl, KDE_vals,
        standard_scale=True):
    """Perform one pyUPMASK outer-loop run."""

    clust_ID = np.array(list(ID))
    clust_xy = np.array(list(xy))

    # Experimental change: scale the data and their errors using the same
    # StandardScaler scale factors so the heteroscedastic GUMM remains in
    # consistent units.
    clust_data, clust_err = prepareData(
        resampleFlag, data, data_err, prfl, standard_scale)

    # This experiment is configured with PCAflag=False.  If PCA is enabled,
    # the transformed full covariance would be needed; until that is supplied,
    # disable heteroscedastic errors after PCA rather than mixing coordinate
    # systems.
    clust_data, clust_err = dimReduc(
        clust_data, clust_err, PCAflag, PCAdims, prfl)

    for _iter in range(IL_runs):
        print("\n IL iteration {}".format(_iter + 1), file=prfl)

        N_clusts, msk_all, N_survived, KDE_vals = inner.loop(
            clust_xy, clust_data, N_membs, N_cl_max, clust_method,
            clRjctMethod, KDE_vals, Kest, C_thresh, cl_method_pars, prfl)

        if N_clusts == N_survived:
            print(" All clusters survived, N={}".format(
                clust_xy.shape[0]), file=prfl)
            break

        if msk_all.sum() < N_membs:
            print(" N_stars<{:.0f} Breaking".format(N_membs), file=prfl)
            break
        print(" A total of {} stars survived in {} clusters".format(
            msk_all.sum(), N_survived), file=prfl)

        clust_ID = clust_ID[msk_all]
        clust_xy = clust_xy[msk_all]
        clust_data = clust_data[msk_all]
        if clust_err is not None:
            clust_err = clust_err[msk_all]

        if GUMM_flag:
            print(" Performing heteroscedastic GUMM analysis...", file=prfl)
            gumm_p = GUMMProbs(clust_data, clust_err)
            prob_cut = GUMMProbCut(GUMM_perc, gumm_p)
            msk = gumm_p > prob_cut

            if msk.sum() > N_membs:
                n_before = len(clust_ID)
                clust_ID = clust_ID[msk]
                clust_xy = clust_xy[msk]
                clust_data = clust_data[msk]
                if clust_err is not None:
                    clust_err = clust_err[msk]
                print(" Rejected {} stars as non-members".format(
                    n_before - msk.sum()), file=prfl)

    if _iter + 1 == IL_runs:
        print("Maximum number of IL runs reached. Breaking", file=prfl)

    # Keep continuous GUMM probabilities for surviving stars so the submitted
    # adaptive KDE bridge receives a probability rather than a purely binary
    # mask. Stars rejected by the inner loop remain zero.
    cl_probs = np.zeros(len(ID), dtype=float)

    if len(clust_ID):
        if GUMM_flag:
            print("Performing final heteroscedastic GUMM analysis...", file=prfl)
            final_gumm = GUMMProbs(clust_data, clust_err)
            prob_cut = GUMMProbCut(GUMM_perc, final_gumm)
            final_gumm = np.where(final_gumm > prob_cut, final_gumm, 0.0)
        else:
            final_gumm = np.ones(len(clust_ID), dtype=float)

        id_to_idx = {st: i for i, st in enumerate(ID)}
        for st, p in zip(clust_ID, final_gumm):
            if st in id_to_idx:
                cl_probs[id_to_idx[st]] = p

    if KDEP_flag:
        print("Performing adaptive spatial KDE analysis...", file=prfl)
        cl_probs = KDEAnalysis.probs(xy, data, cl_probs)

    return list(cl_probs), KDE_vals


def prepareData(resampleFlag, data, data_err, prfl, standard_scale=True):
    """Resample (optional), standardize data, and scale errors consistently."""
    data = np.asarray(data, dtype=float)
    has_err = data_err is not None and np.asarray(data_err).size > 0
    errs = np.asarray(data_err, dtype=float) if has_err else None

    if resampleFlag and errs is not None:
        sampled_data = data + np.random.normal(size=data.shape) * errs
    else:
        sampled_data = np.array(data, copy=True)

    if standard_scale:
        scaler = StandardScaler().fit(sampled_data)
        sampled_data = scaler.transform(sampled_data)
        print("Standard scale: removed mean and scaled to unit variance", file=prfl)
        if errs is not None:
            scale = np.where(scaler.scale_ > 0, scaler.scale_, 1.0)
            errs = errs / scale

    return sampled_data, errs


def dimReduc(cl_data, cl_err, PCAflag, PCAdims, prfl):
    """Optional PCA; heteroscedastic errors are disabled after PCA for now."""
    if PCAflag:
        pca = PCA(n_components=PCAdims)
        cl_data_pca = pca.fit(cl_data).transform(cl_data)
        print(" Selected N={} PCA features".format(PCAdims), file=prfl)
        var_r = ["{:.2f}".format(_) for _ in pca.explained_variance_ratio_]
        print(" Variance ratio: ", ", ".join(var_r), file=prfl)
        if cl_err is not None:
            print(
                " WARNING: heteroscedastic GUMM errors disabled after PCA; "
                "full covariance propagation is not implemented.", file=prfl)
        return cl_data_pca, None
    return cl_data, cl_err


# Backward-compatible helper name retained for external callers.
def reSampleData(resampleFlag, data, data_err, prfl, standard_scale=True):
    return prepareData(
        resampleFlag, data, data_err, prfl, standard_scale)[0]
