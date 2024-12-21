from typing import Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from insitupy._constants import (DEFAULT_CATEGORICAL_CMAP,
                                 DEFAULT_CONTINUOUS_CMAP)
from insitupy.palettes import CustomPalettes


def create_cmap_mapping(data, cmap: Union[str, ListedColormap] = None):

    if cmap is None:
        pal = CustomPalettes()
        cmap = pal.tab20_mod
    try:
        unique_categories = data.cat.categories # in case of categorical pandas series
    except AttributeError:
        try:
            unique_categories = data.categories # in case of numpy categories
        except AttributeError:
            data = np.array(data)
            try:
                unique_categories = np.sort(data[~data.isna()].unique())
            except AttributeError:
                try:
                    unique_categories = np.sort(np.unique(data[~np.isnan(data)]))
                except TypeError:
                    #unique_categories = np.sort(np.unique(data))
                    # Convert all elements to strings before sorting
                    unique_categories = np.sort(np.unique(data.astype(str)))

    # get colormap if necessary
    if not isinstance(cmap, ListedColormap):
        cmap = plt.get_cmap(cmap)

    len_colormap = cmap.N
    category_to_rgba = {category: cmap(i % len_colormap) for i, category in enumerate(unique_categories)}
    return category_to_rgba


def categorical_data_to_rgba(data,
                             cmap: Union[str, ListedColormap],
                             return_mapping: bool = False,
                             nan_val: tuple = (1,1,1,0)
                             ):

    # len_colormap = cmap.N
    # category_to_rgba = {category: cmap(i % len_colormap) for i, category in enumerate(unique_categories)}

    if not isinstance(cmap, dict):
        category_to_rgba = create_cmap_mapping(data, cmap)
    else:
        category_to_rgba = cmap

    if nan_val is not None:
        # add key for nan
        category_to_rgba[np.nan] = nan_val

    res = np.array([category_to_rgba[category] for category in data])

    if return_mapping:
        return res, category_to_rgba
    else:
        return res


def _determine_climits(
    color_values,
    upper_climit_pct,
    lower_climit = None,
    force_above_zero = False
    ) -> list:

    if lower_climit is None:
        lower_climit = color_values.min()

    if force_above_zero:
        color_values_for_calc = color_values[color_values > 0]
    else:
        color_values_for_calc = color_values
    try:
        upper_climit = np.percentile(color_values_for_calc, upper_climit_pct)
    except IndexError:
        # if there were no color values above zero, a IndexError appears
        upper_climit = 0

    climits = [lower_climit, upper_climit]

    return climits


def continuous_data_to_rgba(
    data,
    cmap: Union[str, ListedColormap],
    upper_climit_pct: int = 99,
    lower_climit: Optional[int] = None,
    clip = False,
    nan_val: tuple = (1,1,1,0),
    return_mapping: bool = False
    ):
    if np.any(pd.isna(data)):
        contains_nans = True
        # Convert the numpy array to a pandas Series
        value_series = pd.Series(data)

        # check where there are nas
        isna_mask = value_series.isna()
        notna_mask = ~isna_mask

        # get only values without NaNs
        notna_values = value_series[notna_mask].values
    else:
        notna_values = data
        contains_nans = False

    # get colormap if necessary
    if not isinstance(cmap, ListedColormap):
        cmap = plt.get_cmap(cmap)

    if lower_climit is None:
        lower_climit = np.min(notna_values)

    climits = _determine_climits(color_values=notna_values, upper_climit_pct=upper_climit_pct, lower_climit=lower_climit)

    norm = mpl.colors.Normalize(vmin=climits[0], vmax=climits[1], clip=clip)
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba_data = scalarMap.to_rgba(notna_values)

    if contains_nans:
        result_values = pd.Series(index=value_series.index, dtype="object")
        tuple_list = [tuple(elem) for elem in rgba_data]
        result_values.loc[notna_mask] = tuple_list
        result_values.loc[isna_mask] = [nan_val] * isna_mask.sum()
        result_rgba = np.array(result_values.tolist())
    else:
        result_rgba = rgba_data

    if return_mapping:
        return result_rgba, scalarMap
    else:
        return result_rgba


def _data_to_rgba(
    data,
    continuous_cmap: Union[str, ListedColormap] = DEFAULT_CONTINUOUS_CMAP,
    categorical_cmap: Union[str, ListedColormap] = None,
    upper_climit_pct: int = 99,
    #return_all: bool = False,
    nan_val: tuple = (1,1,1,0),
    ):
    if not is_numeric_dtype(data) or is_bool_dtype(data):
        if is_bool_dtype(data):
            # in case of boolean data it is important to convert it to object type
            # also pandas NAs have to be substituted with numpy NaNs
            data = data.astype('object').replace({pd.NA: np.nan})
        if categorical_cmap is None:
            # pal = CustomPalettes()
            # categorical_cmap = pal.tab20_mod
            categorical_cmap = DEFAULT_CATEGORICAL_CMAP
        rgba_list, mapping = categorical_data_to_rgba(data=data, cmap=categorical_cmap,
                                        return_mapping=True,
                                        nan_val=nan_val)
        cmap = categorical_cmap
    else:
        rgba_list, mapping = continuous_data_to_rgba(data=data, cmap=continuous_cmap,
                                       upper_climit_pct=upper_climit_pct,
                                       return_mapping=True)
        cmap = continuous_cmap

    return rgba_list, mapping, cmap