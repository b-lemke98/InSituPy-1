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
        pixelwidth_per_subplot: int = 200,
        raw: bool = False,
        layer: Optional[str] = None,
        obsm_key: str = 'spatial',
        origin_zero: bool = False, # whether to start axes ticks at 0
        xlim: Optional[Tuple[int, int]] = None,
        ylim: Optional[Tuple[int, int]] = None,
        spot_size: float = 10,
        histogram_setting: Optional[Union[Literal["auto"], Tuple[int, int]]] = "auto"
        ):

        # add arguments to object
        self.key = key
        self.spot_size = spot_size

        # convert limits to list
        self.xlim = list(xlim) if xlim is not None else xlim
        self.ylim = list(ylim) if ylim is not None else ylim

        ## Extract coordinates
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
            # xmin = np.min([self.x_coords.min(), self.y_coords.min()]) # make sure that result is always a square
            # xmax = np.max([self.x_coords.max(), self.y_coords.max()])
            xmin = self.x_coords.min()
            xmax = self.x_coords.max()
            print(xmin)
            print(xmax)

            # include margin
            #self.xlim = (xmin - spot_size, xmax + spot_size)
            self.xlim = (xmin, xmax)

        if self.ylim is None:
            # ymin = np.min([self.x_coords.min(), self.y_coords.min()])
            # ymax = np.max([self.x_coords.max(), self.y_coords.max()])
            ymin = self.y_coords.min()
            ymax = self.y_coords.max()

            # include margin
            #self.ylim = (ymin - spot_size, ymax + spot_size)
            self.ylim = (ymin, ymax)

        # extract image information
        if ImageDataObject is not None:
            # pick the image with the right resolution for plotting
            max_pixel_size = np.max([self.xlim[1] - self.xlim[0], self.ylim[1] - self.ylim[0]]) / pixelwidth_per_subplot
            orig_pixel_size = ImageDataObject.metadata[image_key]["pixel_size"]
            img_pyramid = ImageDataObject[image_key]
            pixel_sizes_levels = np.array([orig_pixel_size * (2**i) for i in range(len(img_pyramid))])

            try:
                selected_level = np.where(pixel_sizes_levels <= max_pixel_size)[0][-1].item()
                selected_pixel_size = pixel_sizes_levels[selected_level].item()
            except IndexError:
                selected_level = 0
                selected_pixel_size = pixel_sizes_levels[selected_level].item()

            # extract parameters from ImageDataObject
            self.pixel_size = selected_pixel_size
            self.image = img_pyramid[selected_level]

            ywidth = self.image.shape[0]
            xwidth = self.image.shape[1]

            # crop image
            # self.pixel_xlim = [int(elem / selected_pixel_size) for elem in self.xlim]
            # self.pixel_ylim = [int(elem / selected_pixel_size) for elem in self.ylim]
            self.pixel_xlim = [int(elem / selected_pixel_size) if int(elem / selected_pixel_size) <= xwidth else xwidth for elem in self.xlim]
            self.pixel_ylim = [int(elem / selected_pixel_size) if int(elem / selected_pixel_size) <= ywidth else ywidth for elem in self.ylim]

            self.image = self.image[
                self.pixel_ylim[0]:self.pixel_ylim[1],
                self.pixel_xlim[0]:self.pixel_xlim[1]
                ]

            if histogram_setting is None:
                self.vmin = self.vmax = None
            elif histogram_setting == "auto":
                self.vmin = da.percentile(self.image.ravel(), 30).compute().item()
                self.vmax = da.percentile(self.image.ravel(), 99.5).compute().item()
            elif isinstance(histogram_setting, tuple):
                self.vmin = histogram_setting[0]
                self.vmax = histogram_setting[1]
            else:
                raise ValueError(f"Unknown type for histogram_setting: {type(histogram_setting)}")

        else:
            self.image = None


        # get color values for expression data or categories
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