import anndata
import dask.array as da
import zarr
from pandas.api.types import is_numeric_dtype, is_string_dtype
from scipy.sparse import issparse
import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any, Literal
import os
from pathlib import Path
import json

class textformat:
    '''
    Helper class to format printed text.
    e.g. print(color.RED + color.BOLD + 'Hello, World!' + color.END)
    '''
    # colors and formats
    # PURPLE = '\033[95m'
    # CYAN = '\033[96m'
    # DARKCYAN = '\033[36m'
    # BLUE = '\033[94m'
    # GREEN = '\033[92m'
    # YELLOW = '\033[93m'
    # RED = '\033[91m'
    # BOLD = '\033[1m'
    # UNDERLINE = '\033[4m'
    # END = '\033[0m'
    
    ResetAll = "\033[0m"

    Bold       = "\033[1m"
    Dim        = "\033[2m"
    Underlined = "\033[4m"
    Blink      = "\033[5m"
    Reverse    = "\033[7m"
    Hidden     = "\033[8m"

    ResetBold       = "\033[21m"
    ResetDim        = "\033[22m"
    ResetUnderlined = "\033[24m"
    ResetBlink      = "\033[25m"
    ResetReverse    = "\033[27m"
    ResetHidden     = "\033[28m"

    Default      = "\033[39m"
    Black        = "\033[30m"
    Red          = "\033[31m"
    Green        = "\033[32m"
    Yellow       = "\033[33m"
    Blue         = "\033[34m"
    Magenta      = "\033[35m"
    Cyan         = "\033[36m"
    LightGray    = "\033[37m"
    DarkGray     = "\033[90m"
    LightRed     = "\033[91m"
    LightGreen   = "\033[92m"
    LightYellow  = "\033[93m"
    LightBlue    = "\033[94m"
    Purple       = "\033[95m"
    LightCyan    = "\033[96m"
    White        = "\033[97m"

    BackgroundDefault      = "\033[49m"
    BackgroundBlack        = "\033[40m"
    BackgroundRed          = "\033[41m"
    BackgroundGreen        = "\033[42m"
    BackgroundYellow       = "\033[43m"
    BackgroundBlue         = "\033[44m"
    BackgroundMagenta      = "\033[45m"
    BackgroundCyan         = "\033[46m"
    BackgroundLightGray    = "\033[47m"
    BackgroundDarkGray     = "\033[100m"
    BackgroundLightRed     = "\033[101m"
    BackgroundLightGreen   = "\033[102m"
    BackgroundLightYellow  = "\033[103m"
    BackgroundLightBlue    = "\033[104m"
    BackgroundLightMagenta = "\033[105m"
    BackgroundLightCyan    = "\033[106m"
    BackgroundWhite        = "\033[107m"

    # signs
    TSIGN = "\u251c"
    LSIGN = "\u2514"
    HLINE = "\u2500"
    RARROWHEAD = "\u27A4"
    TICK = "\u2714"
    CIRCLE_EMPTY = "\u25EF"
    CIRCLE_FILLED = "\u2B24"
    CIRCLE_HALF = "\u25D0"
    CIRCLE_THREEQUARTER = "\u25D5"
    CIRCLE_ONEQUARTER = "\u25D4"

    # spacer
    SPACER = "    "
    
def remove_last_line_from_csv(filename):
    with open(filename) as myFile:
        lines = myFile.readlines()
        last_line = lines[len(lines)-1]
        lines[len(lines)-1] = last_line.rstrip()
    with open(filename, 'w') as myFile:    
        myFile.writelines(lines)
        
def decode_robust(s, encoding="utf-8"):
    try:
        return s.decode(encoding)
    except (UnicodeDecodeError, AttributeError):
        return s
    
def decode_robust_series(s, encoding="utf-8"):
    '''
    Function to decode a pandas series in a robust fashion with different checks.
    This circumvents the return of NaNs and makes a decision in case of different errors.
    '''
    if is_numeric_dtype(s):
        return s
    if is_string_dtype(s):
        return s
    try:
        decoded = s.str.decode(encoding)
        if decoded.isna().all():
            return decoded
        elif decoded.isna().any():
            namask = decoded.isna()
            decoded[namask] = s[namask]
        return decoded
    except (UnicodeDecodeError, AttributeError):
        return s
    
def convert_to_list(elem):
    '''
    Return element to list if it is not a list already.
    '''
    return [elem] if isinstance(elem, str) else list(elem)

# checker functions for data sanity
def check_adata(adata):
    if type(adata) is not anndata.AnnData:
        raise TypeError('Input is not a valid AnnData object')


def check_batch(batch, obs, verbose=False):
    if batch not in obs:
        raise ValueError(f'column {batch} is not in obs')
    elif verbose:
        print(f'Object contains {obs[batch].nunique()} batches.')


def check_hvg(hvg, hvg_key, adata_var):
    if type(hvg) is not list:
        raise TypeError('HVG list is not a list')
    else:
        if not all(i in adata_var.index for i in hvg):
            raise ValueError('Not all HVGs are in the adata object')
    if not hvg_key in adata_var:
        raise KeyError('`hvg_key` not found in `adata.var`')

def check_sanity(adata, batch, hvg, hvg_key):
    check_adata(adata)
    check_batch(batch, adata.obs)
    if hvg:
        check_hvg(hvg, hvg_key, adata.var)


def split_batches(adata, batch, hvg=None, return_categories=False):
    split = []
    batch_categories = adata.obs[batch].unique()
    if hvg is not None:
        adata = adata[:, hvg]
    for i in batch_categories:
        split.append(adata[adata.obs[batch] == i].copy())
    if return_categories:
        return split, batch_categories
    return split
    
def check_raw(X):
    '''
    Check if a matrix consists of raw integer counts or if it is processed already.
    '''
    
    # convert sparse matrix to numpy array
    if issparse(X):
        X = X.toarray()
    
    # check if the matrix contains raw counts
    if not np.all(np.modf(X)[0] == 0):
        raise ValueError("Anndata object does not contain raw counts. Preprocessing aborted.")

## read utils

def load_pyramid(store):
    '''
    Function to load pyramid.
    From: https://www.youtube.com/watch?v=8TlAAZcJnvA
    '''
    # Open store (root group)
    grp = zarr.open(store, mode='r')

    # Read multiscale metadata
    datasets = grp.attrs["multiscales"][0]["datasets"]

    return [
        da.from_zarr(store, component=d["path"])
        for d in datasets
    ]
    
def read_json(
    file: Union[str, os.PathLike, Path],
    ) -> dict:
    '''
    Function to load json files as dictionary.
    '''
    # load metadata file
    with open(file, "r") as metafile:
        metadata = json.load(metafile)
        
    return metadata
