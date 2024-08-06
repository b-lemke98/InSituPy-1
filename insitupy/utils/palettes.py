from typing import Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, rgb2hex
from pandas.api.types import is_numeric_dtype


class CustomPalettes:
    '''
    Class containing a collection of custom color palettes.
    '''
    def __init__(self):
        # palette for colorblind people. From: https://gist.github.com/thriveth/8560036
        self.colorblind = ListedColormap(
            ['#377eb8', '#ff7f00', '#4daf4a',
             '#f781bf', '#dede00', '#a65628',
             '#984ea3', '#999999', '#e41a1c'], name="colorblind")

        # palette from Caro. Optimized for colorblind people.
        self.caro = ListedColormap(['#3288BD','#440055', '#D35F5F', '#A02C2C','#225500', '#66C2A5', '#447C69'], name="caro")

        # from https://thenode.biologists.com/data-visualization-with-flying-colors/research/
        self.okabe_ito = ListedColormap(["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"], name="okabe_ito")
        self.tol_bright = ListedColormap(["#EE6677", "#228833", "#4477AA", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"], name="tol_bright")
        self.tol_muted = ListedColormap(["#88CCEE", "#44AA99", "#117733", "#332288", "#DDCC77", "#999933", "#CC6677", "#882255", "#AA4499", "#DDDDDD"], name="tol_muted")
        self.tol_light = ListedColormap(["#BBCC33", "#AAAA00", "#77AADD", "#EE8866", "#EEDD88", "#FFAABB", "#99DDFF", "#44BB99", "#DDDDDD"], name="tol_light")

        # generate modified tab20 color palette
        colormap = mpl.colormaps["tab20"]

        # split by high intensity and low intensity colors in tab20
        cmap1 = colormap.colors[::2]
        cmap2 = colormap.colors[1::2]

        # concatenate color cycle
        color_cycle = cmap1[:7] + cmap1[8:] + cmap2[:7] + cmap2[8:]
        self.tab20_mod = ListedColormap([rgb2hex(elem) for elem in color_cycle])


    def show_all(self):
        '''
        Plots all colormaps in the collection.
        '''
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        # get list of names and respective
        cmaps = []
        names = []
        for name, cmap in vars(self).items():
            if isinstance(cmap, ListedColormap):
                cmaps.append(cmap)
                names.append(name)

        # Create figure and adjust figure height to number of colormaps
        nrows = len(vars(self).values())
        figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
        fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
        fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                            left=0.2, right=0.99)

        axs = axs.ravel()

        for ax, name, cmap in zip(axs, names, cmaps):
            ax.imshow(gradient, aspect='auto', cmap=cmap)
            ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
                    transform=ax.transAxes)

        # Turn off *all* ticks & spines, not just the ones with colormaps.
        for ax in axs:
            ax.set_axis_off()


def cmap2hex(cmap):
    '''
    Generate list of hex-coded colors from cmap.
    '''
    hexlist = [rgb2hex(cmap(i)) for i in range(cmap.N)]
    return hexlist


def _determine_climits(
    color_values,
    upper_climit_pct,
    lower_climit = None
    ) -> list:

    if lower_climit is None:
        lower_climit = color_values.min()

    color_values_above_zero = color_values[color_values > 0]
    try:
        upper_climit = np.percentile(color_values_above_zero, upper_climit_pct)
    except IndexError:
        # if there were not color_values_above_zero, a IndexError appears
        upper_climit = 0

    climits = [lower_climit, upper_climit]

    return climits


def _determine_color_settings(
    color_values,
    cmap,
    upper_climit_pct
    ):
    # check if the data should be plotted categorical or continous
    if is_numeric_dtype(color_values):
        is_categorical = False # if the data is numeric it should be plotted continous
    else:
        is_categorical = True # if the data is not numeric it should be plotted categorically

    if is_categorical:
        # get color cycle for categorical data
        color_mode = "cycle"
        palettes = CustomPalettes()
        color_cycle = getattr(palettes, "tab20_mod").colors
        color_map = None
        climits = None
    else:
        color_mode = "colormap"
        color_map = cmap
        color_cycle = None

        climits = _determine_climits(
            color_values=color_values,
            upper_climit_pct=upper_climit_pct
        )

    return color_mode, color_cycle, color_map, climits

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
            try:
                unique_categories = np.sort(data[~data.isna()].unique())
            except AttributeError:
                try:
                    unique_categories = np.sort(np.unique(data[~np.isnan(data)]))
                except TypeError:
                    unique_categories = np.sort(np.unique(data))

    # get colormap if necessary
    if not isinstance(cmap, ListedColormap):
        cmap = plt.get_cmap(cmap)

    len_colormap = cmap.N
    category_to_rgba = {category: cmap(i % len_colormap) for i, category in enumerate(unique_categories)}
    return category_to_rgba

def continuous_data_to_rgba(
    data,
    cmap: Union[str, ListedColormap],
    upper_climit_pct: int = 99,
    lower_climit: Optional[int] = 0,
    clip = False
    ):
    # get colormap if necessary
    if not isinstance(cmap, ListedColormap):
        cmap = plt.get_cmap(cmap)

    climits = _determine_climits(color_values=data, upper_climit_pct=upper_climit_pct, lower_climit=lower_climit)

    norm = mpl.colors.Normalize(vmin=climits[0], vmax=climits[1], clip=clip)
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)
    return scalarMap.to_rgba(data)

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

def data_to_rgba(
    data,
    continuous_cmap: Union[str, ListedColormap] = "viridis",
    categorical_cmap: Union[str, ListedColormap] = None,
    upper_climit_pct: int = 99,
    return_mapping: bool = False,
    nan_val: tuple = (1,1,1,0)
    ):
    if is_numeric_dtype(data):
        return continuous_data_to_rgba(data=data, cmap=continuous_cmap, upper_climit_pct=upper_climit_pct)
    else:
        if categorical_cmap is None:
            pal = CustomPalettes()
            categorical_cmap = pal.tab20_mod
        return categorical_data_to_rgba(data=data, cmap=categorical_cmap,
                                        return_mapping=return_mapping,
                                        nan_val=nan_val)


