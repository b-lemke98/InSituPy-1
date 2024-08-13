from math import sqrt
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
from scipy.spatial import distance as dist
from skmisc.loess import loess

#from sympy import lowergamma
#from ..exceptions import ImportErrorLoess, ModuleNotFoundOnWindows
from ._regression import bootstrap_loess, lowess


def smooth_fit(xs: np.ndarray, ys: np.ndarray,
    #x_to_fit_on: Optional[List] = None,
    #dist_thrs: Optional[float] = None,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    #stepsize: Optional[float] = None,
    nsteps: Optional[float] = None,
    method: Literal["lowess", "loess"] = "loess",
    stderr: bool = True,
    loess_bootstrap: bool = True,
    K: int = 100
    ):

    """Smooth curve using loess

    will perform curve fitting using skmisc.loess,
    points above 'dist_thrs' will be excluded.
    Parameters:
    ----------
    xs : np.ndarray
        x values
    ys : np.ndarray
        y values
    min: Optional[float]
        exclude (x,y) tuples were x < min
    max: Optional[float]
        exclude (x,y) tuples were x > max
    nsteps = Optional[float]
        Number of steps x is divided into for the prediction.
    method: str
        Which method to use for the smoothing. Options: "loess" or "lowess".
        "loess": Uses `skmisc.loess`. This is not implemented for Windows.
        "lowess": Uses `statsmodels.nonparametric.smoothers_lowess.sm_lowess` which is available on all tested platforms.
    stderr: bool
        Whether to calculate standard deviation of the prediction. Depends on the method used:
        "loess": Standard deviation returned by package is used.
        "lowess" Bootstrapping is used to estimate a confidence interval.
    K: int
        Only needed for `method="lowess"`. Determines number of bootstrapping cycles.

    Returns:
    -------
    A tuple with included x and y-values (xs',ys'), as well
    as fitted y-values (ys_hat) together with associated
    standard errors. The tuple is on the form
    (xs',ys',y_hat,std_err)

    From: https://github.com/almaan/ST-mLiver
    """

    # check method
    if method == "loess":
        use_loess = True
        if (stderr == True) & (bootstrap_loess == False):
            warn("When using the LOESS method and `stderr=True`, `bootstrap_loess` should be set True. Otherwise it could be that the kernel is crashing due to the high number of data points in Xenium experiments.")
    elif method == "lowess":
        use_loess = False
    else:
        raise ValueError('Invalid `method`. Expected is one of: ["loess", "lowess"')

    # sort x values
    srt = np.argsort(xs)
    xs = xs[srt]
    ys = ys[srt]

    # determine min and max x values and select x inside this range
    if xmin is None:
        xmin = xs.min()
    if xmax is None:
        xmax = xs.max()

    keep = (xs >= xmin) & (xs <= xmax)
    xs = xs[keep]
    ys = ys[keep]

    # generate loess class object
    if use_loess:
        ls = loess(xs, ys)
    else:
        ls = lowess(xs, ys)

    # fit loess class to data
    ls.fit()

    # if stepsize is given determine xs to fit the data on
    if nsteps is not None:
        stepsize = (xmax - xmin) / nsteps
        #if stepsize is not None:
        xs_pred = np.asarray(np.arange(xmin, xmax+stepsize, stepsize))
        xs_pred = np.linspace(xmin, xmax, nsteps)
        xs_pred = xs_pred[(xs_pred < xs.max()) & (xs_pred > xs.min())]

    # predict on data
    if use_loess:
        if loess_bootstrap:
            pred =  ls.predict(xs_pred, stderror=False)
        else:
            pred =  ls.predict(xs_pred, stderror=stderr)
    else:
        pred =  ls.predict(xs_pred, stderror=stderr, K=K)

    # get predicted values
    ys_hat = pred.values

    if stderr:
        # calculate confidence intervals and standard error if that was not calculated before
        if loess_bootstrap:
            bl = bootstrap_loess(ls)
            bl.calc_loess_stderror_by_bootstrap(newdata=xs_pred, K=K)
            conf = bl.confidence()
        else:
            stderr = pred.stderr
            conf = pred.confidence()

        # retrieve upper and lower confidence intervals
        lower = conf.lower
        upper = conf.upper
    else:
        lower = np.nan
        upper = np.nan

    df = pd.DataFrame({
        'x': xs_pred,
        #'y': ys,
        'y_pred': ys_hat,
        'std': stderr,
        'conf_lower': lower,
        'conf_upper': upper
    })

    #return (xs,ys,ys_hat,stderr)
    return df