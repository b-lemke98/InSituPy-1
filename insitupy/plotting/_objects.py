import pickle
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import dask.array as da
import numpy as np
from anndata import AnnData
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from insitupy._core._checks import check_raw
from insitupy._core.dataclasses import ImageData


class _ConfigSpatialPlot:
    '''
    Object extracting spatial coordinates and expression data from anndata object.
    '''
    def __init__(
        self,
        adata: AnnData,
        key: List[str],
        ImageDataObject: Optional[ImageData],
        image_key: Optional[str] = None,
        image_pyramid_level: int = 3,
        raw: bool = False,
        layer: Optional[str] = None,
        obsm_key: str = 'spatial',
        origin_zero: bool = True, # whether to start axes ticks at 0
        xlim: Optional[Tuple[int, int]] = None,
        ylim: Optional[Tuple[int, int]] = None,
        spot_size: float = 10,
        margin: bool = False # whether to leave margin of one spot width around the plot
        ):

        # add arguments to object
        self.key = key
        self.spot_size = spot_size

        # convert limits to list
        self.xlim = list(xlim) if xlim is not None else xlim
        self.ylim = list(ylim) if ylim is not None else ylim


        ## Extract coordinates
        if ImageDataObject is not None:
            # extract parameters from ImageDataObject
            self.pixel_size = ImageDataObject.metadata[image_key]["pixel_size"] * (2**image_pyramid_level)
            self.image = ImageDataObject[image_key][image_pyramid_level]
        else:
            self.image = None

        # extract x and y pixel coordinates and convert to micrometer
        self.x_coords = adata.obsm[obsm_key][:, 0].copy()
        self.y_coords = adata.obsm[obsm_key][:, 1].copy()

        # shift coordinates that they start at (0,0)
        if origin_zero:
            self.x_offset = self.x_coords.min()
            self.y_offset = self.y_coords.min()
            self.x_coords -= self.x_offset
            self.y_coords -= self.y_offset
        else:
            self.x_offset = self.y_offset = 0

        if self.xlim is None:
            xmin = np.min([self.x_coords.min(), self.y_coords.min()]) # make sure that result is always a square
            xmax = np.max([self.x_coords.max(), self.y_coords.max()])

            self.xlim = (xmin - spot_size, xmax + spot_size)
        elif margin:
            self.xlim[0] -= spot_size
            self.xlim[1] += spot_size

        if self.ylim is None:
            ymin = np.min([self.x_coords.min(), self.y_coords.min()])
            ymax = np.max([self.x_coords.max(), self.y_coords.max()])

            self.ylim = (ymin - spot_size, ymax + spot_size)
        elif margin:
            self.ylim[0] -= spot_size
            self.ylim[1] += spot_size

        self.color_values, self.categorical = _extract_color_values(
            adata=adata, key=self.key, raw=raw, layer=layer
        )

def _extract_color_values(adata, key, raw, layer):
    ## Extract expression data
    # check if plotting raw data
    adata_X, adata_var, adata_var_names = check_raw(
        adata,
        use_raw=raw,
        layer=layer
        )

    # locate gene in matrix and extract values
    if key in adata_var_names:
        idx = adata_var.index.get_loc(key)
        color_values = adata_X[:, idx].copy()
        categorical = False

    elif key in adata.obs.columns:
        color_values = adata.obs[key].values
        if is_numeric_dtype(adata.obs[key]):
            categorical = False
        else:
            categorical = True
    else:
        color_values = None
        categorical = None

    return color_values, categorical