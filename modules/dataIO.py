import numpy as np
from pathlib import Path
from astropy.io import ascii
from astropy.table import Column
from astropy.table import Table, vstack
import configparser
import warnings
from distutils.util import strtobool
from sklearn.preprocessing import MinMaxScaler
from .outlierRjct import stdRegion, sklearnMethod


def readINI():
    """Read .ini config file."""

    def vtype(var):
        tp, v = var.split('_', 1)
        if tp == 'int':
            return int(v)
        elif tp == 'float':
            return float(v)
        elif tp == 'bool':
            return bool(strtobool(v))
        elif tp == 'str':
            return v
        raise ValueError("Unknown parameter type '{}'".format(tp))

    in_params = configparser.ConfigParser()
    in_params.read('params.ini')

    gen_pars = in_params["General parameters"]
    rnd_seed, verbose, parallel_flag, parallel_procs, np_mthread =\
        gen_pars.get('rnd_seed'), gen_pars.getint('verbose'),\
        gen_pars.getboolean('parallel'), gen_pars.get('processes'),\
        gen_pars.getboolean('numpy_multi')
    if parallel_procs not in ('None', 'none', 'NONE'):
        parallel_procs = int(parallel_procs)
        if parallel_procs <= 0:
            raise ValueError("The 'processes' parameter must be >0")

    data_columns = in_params["Input file's data columns"]
    ID_c = data_columns['ID']
    x_c, y_c = data_columns['xy_coords'].split()
    data_cols = data_columns['data'].split()
    oultr_method = data_columns.get('oultr_method')
    stdRegion_nstd = data_columns.getfloat('stdRegion_nstd')

    outer_loop = in_params['Outer loop']
    OL_runs, resampleFlag, PCAflag, PCAdims, GUMM_flag, KDEP_flag =\
        outer_loop.getint('OL_runs'), outer_loop.getboolean('resampleFlag'),\
        outer_loop.getboolean('PCAflag'), outer_loop.getint('PCAdims'),\
        outer_loop.getboolean('GUMM_flag'), outer_loop.getboolean('KDEP_flag')
    GUMM_perc = outer_loop.get('GUMM_perc')
    if GUMM_perc != 'auto':
        GUMM_perc = float(GUMM_perc)

    # Experimental change: always read uncertainty columns when they are
    # configured.  The heteroscedastic GUMM consumes them even when
    # resampleFlag=False.
    data_errs = []
    if data_columns.get('uncert', fallback='').strip():
        data_errs = data_columns['uncert'].split()

    inner_loop = in_params['Inner loop']
    IL_runs, N_membs, N_cl_max, clust_method = inner_loop.getint('IL_runs'),\
        inner_loop.getint('N_membs'), inner_loop.getint('N_cl_max'),\
        inner_loop.get('clust_method')

    allowed_clust_methods = (
        'KMeans', 'MiniBatchKMeans', 'AffinityPropagation', 'MeanShift',
        'SpectralClustering', 'AgglomerativeClustering', 'DBSCAN', 'OPTICS',
        'Birch', 'GaussianMixture', 'BayesianGaussianMixture', 'Voronoi',
        'rkmeans', 'kNNdens', 'HDBSCAN')
    if clust_method not in allowed_clust_methods:
        raise ValueError("Unrecognized clustering method '{}'".format(
            clust_method))

    single_run_methods = ('Voronoi', 'rkmeans', 'kNNdens')
    if clust_method in single_run_methods and OL_runs > 1\
            and resampleFlag is False:
        warnings.warn(
            "Single run method selected, only one OL run will be processed")
        OL_runs = 1

    clRjctMethod, C_thresh = 'rkfunc', 1.

    cl_method_pars = {}
    for key, val in in_params['Clustering parameters'].items():
        cl_method_pars[key] = vtype(val)

    return [
        np_mthread, parallel_flag, parallel_procs, rnd_seed, verbose,
        ID_c, x_c, y_c, data_cols, data_errs, oultr_method, stdRegion_nstd,
        OL_runs, resampleFlag, PCAflag, PCAdims, GUMM_flag, GUMM_perc,
        KDEP_flag, IL_runs, N_membs, N_cl_max, clust_method, clRjctMethod,
        C_thresh, cl_method_pars]


def readFiles():
    files = []
    for pp in Path('input').iterdir():
        if not pp.name.endswith(".md"):
            if pp.is_file():
                files += [pp]
            else:
                files += [arch for arch in pp.iterdir()]
    return files


def dread(file_path, ID_c, x_c, y_c, data_cols, data_errs):
    data = Table.read(file_path, format='ascii')
    N_d = len(data)
    print("Stars read         : {}".format(N_d))

    try:
        required = list(data_cols) + list(data_errs)
        msk = np.logical_and.reduce([~data[_].mask for _ in required])
        data_rjct = data[~msk]
        data = data[msk]
        print("Stars removed      : {}".format(N_d - len(data)))
    except AttributeError:
        data_rjct = []

    if ID_c == 'None':
        N_d = len(data)
        ID_data = np.arange(1, N_d + 1)
    else:
        ID_data = data[ID_c]
    xy_data = np.array([data[x_c], data[y_c]]).T
    cl_data = np.array([data[_] for _ in data_cols]).T

    cl_errs = np.array([])
    if data_errs:
        cl_errs = np.array([data[_] for _ in data_errs]).T
    print("Data dimensions    : {}".format(cl_data.shape[1]))

    return data, ID_data, xy_data, cl_data, cl_errs, data_rjct


def dmask(ID, xy, pdata, perrs, oultr_method, stdRegion_nstd):
    if oultr_method == 'stdregion':
        msk_data = stdRegion(pdata, stdRegion_nstd)
    else:
        msk_data = sklearnMethod(pdata, oultr_method)

    ID_data, xy_data, cl_data = ID[msk_data], xy[msk_data], pdata[msk_data]

    if perrs.size:
        data_err = perrs[msk_data]
    else:
        data_err = np.array([])

    print("Masked outliers    : {}".format((~msk_data).sum()))
    if oultr_method == 'stdregion':
        print(" N_std             : {}".format(stdRegion_nstd))

    return msk_data, ID_data, xy_data, cl_data, data_err


def dxynorm(xy_data):
    _xrange, _yrange = np.ptp(xy_data, 0)
    perc_sq = abs(1. - _xrange / _yrange)
    if perc_sq > .05:
        print((
            "WARNING: (x, y) frame deviates from a square region "
            "by {:.0f}%").format(perc_sq * 100.))
    xy = MinMaxScaler().fit(xy_data).transform(xy_data)
    print("Coordinates scaled : [0, 1]")
    return xy


def dwrite(out_folder, file_path, full_data, msk_data, data_rjct, probs_mean):
    out_path = Path(out_folder, *file_path.parts[1:])

    pf = np.zeros(len(full_data)) - 1.
    pf[msk_data] = probs_mean
    full_data.add_column(Column(np.round(pf, 4), name='probs_final'))

    if len(data_rjct) > 0:
        pf = np.zeros(len(data_rjct)) - 1.
        data_rjct.add_column(Column(pf), name='probs_final')
        full_data = vstack([full_data, data_rjct])

    ascii.write(full_data, out_path, overwrite=True)
