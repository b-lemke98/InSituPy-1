import gc
import math
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from insitupy._core._utils import _get_cell_layer
from insitupy._core.insitudata import InSituData
from insitupy._core.insituexperiment import InSituExperiment
from insitupy.io.plots import save_and_show_figure
from insitupy.plotting._colors import create_cmap_mapping, get_crange
from insitupy.plotting._objects import _ConfigSpatialPlot, _ImageData
from insitupy.utils._adata import _extract_groups
from insitupy.utils.utils import convert_to_list, remove_empty_subplots


class MultiSpatialPlot:
    '''
    Class to render scatter plots of single-cell spatial transcriptomics data.
    '''
    def __init__(self,
                 data,
                 keys: Union[str, List[str]],
                 cells_layer: Optional[str] = None,
                 data_ids: Optional[List[int]] = None,
                #  groupby: str = 'id',
                #  groups: Optional[List[str]] = None,
                 raw: bool = False,
                 layer: Optional[str] = None,
                 fig: plt.figure = None,
                 ax: plt.Axes = None,
                 max_cols: int = 4,
                 xlim: Optional[Tuple[float, float]] = None,
                 ylim: Optional[Tuple[float, float]] = None,
                 normalize_crange_not_for: List = [],
                 crange: Optional[List[int]] = None,
                 crange_type: Literal['minmax', 'percentile'] = 'minmax',
                 palette: str = 'tab10',
                 cmap_center: Optional[float] = None,
                 dpi_display: int = 80,
                 obsm_key: str = 'spatial',
                 origin_zero: bool = True,
                 spot_size: float = 10,
                 margin: bool = True,
                 spot_type: str = 'o',
                 cmap: str = 'viridis',
                 background_color: str = 'white',
                 alpha: float = 1,
                 colorbar: bool = True,
                 clb_title: Optional[str] = None,
                 header: Optional[str] = None,

                 # image stuff
                 image_key: Optional[str] = None,
                 image_pyramid_level: int = 3,
                 histogram_setting: Optional[Tuple[int, int]] = None,
                 lowres: bool = True,

                 # saving
                 savepath: Optional[str] = None,
                 save_only: bool = False,
                 dpi_save: int = 300,
                 show: bool = True,

                 # less important
                 prefix_groups: str = '',
                 groupheader_fontsize: int = 20
                 ):
        assert isinstance(data, InSituExperiment) or isinstance(data, InSituData), "`data` must be either InSituData or InSituExperiment."

        #self.adata = adata
        self.data = data
        self.keys = keys
        self.cells_layer = cells_layer
        self.data_ids = data_ids
        # self.groupby = groupby
        # self.groups = groups
        self.raw = raw
        self.layer = layer
        self.fig = fig
        self.ax = ax
        self.max_cols = max_cols
        self.xlim = xlim
        self.ylim = ylim
        self.normalize_crange_not_for = normalize_crange_not_for
        self.crange = crange
        self.crange_type = crange_type
        self.palette = palette
        self.cmap_center = cmap_center
        self.dpi_display = dpi_display
        self.obsm_key = obsm_key
        self.origin_zero = origin_zero,
        self.spot_size = spot_size
        self.margin = margin
        self.spot_type = spot_type
        self.cmap = cmap
        self.background_color = background_color
        self.alpha = alpha
        self.colorbar = colorbar
        self.clb_title = clb_title
        self.header = header
        self.savepath = savepath
        self.save_only = save_only
        self.dpi_save = dpi_save
        self.show = show

        # image stuff
        self.image_key = image_key
        self.image_pyramid_level = image_pyramid_level
        self.histogram_setting = histogram_setting
        self.lowres = lowres

        # less important
        self.prefix_groups = prefix_groups
        self.groupheader_fontsize = groupheader_fontsize

        # check arguments
        self.check_arguments()

        # plotting
        if self.ax is None:
            self.setup_subplots()
        else:
            assert self.fig is not None, "If axis for plotting is given, also a figure object needs to be provided via `fig`"
            assert len(self.keys) == 1, "If single axis is given not more than one key is allowed."

        self.plot_to_subplots()

        save_and_show_figure(
            savepath=self.savepath,
            fig=self.fig,
            save_only=self.save_only,
            show=self.show,
            dpi_save=self.dpi_save
            )

        gc.collect()

    def check_arguments(self):
        #

        # convert arguments to lists
        self.keys = convert_to_list(self.keys)
        # self.keys = [self.keys] if isinstance(self.keys, str) else list(self.keys)
        # if self.xlim is not None:
        #     self.xlim = [self.xlim] if isinstance(self.xlim, str) else list(self.xlim)
        # if self.xlim is not None:
        #     self.ylim = [self.ylim] if isinstance(self.ylim, str) else list(self.ylim)

        # check if cmap is supposed to be centered
        if self.cmap_center is None:
            self.normalize=None
        else:
            self.normalize = colors.CenteredNorm(vcenter=self.cmap_center)

        # set multiplot variables
        self.multikeys = False
        self.multigroups = False
        if len(self.keys) > 1:
            self.multikeys = True

        try:
            self.n_data = len(self.data)
        except TypeError:
            # if the data is an InSituData, it raises a TypeError
            self.n_data = 1

        if self.n_data > 1:
            self.multigroups = True
        elif self.n_data == 1:
            self.multigroups = False
        else:
            raise ValueError(f"n_data < 1: {self.n_data}")

        # if self.groupby is None:
        #     self.groups = [None]
        # else:
        #     if self.groups is None:
        #         self.groups = list(self.adata.obs[self.groupby].unique())
        #     else:
        #         self.groups = [self.groups] if isinstance(self.groups, str) else list(self.groups)
        #     if len(self.groups) > 1:
        #         self.multigroups = True

        # # determine the color range for each key
        # self.crange_per_key_dict = {
        #     key: get_crange(
        #         self.adata,
        #         #self.groupby, self.groups,
        #         key=key,
        #         use_raw=self.raw, layer=self.layer,
        #         ctype=self.crange_type
        #         ) if key not in self.normalize_crange_not_for else None for key in self.keys
        #     }


    def setup_subplots(self):
        if self.multigroups:
            if self.multikeys:
                n_rows = self.n_data
                self.max_cols = len(self.keys)
                n_plots = n_rows * self.max_cols
                self.fig, self.axs = plt.subplots(n_rows, self.max_cols,
                                                  figsize=(7.6 * self.max_cols, 6 * n_rows),
                                                  dpi=self.dpi_display)
                self.fig.tight_layout() # helps to equalize size of subplots. Without the subplots change parameters during plotting which results in differently sized spots.

            else:
                n_plots = self.n_data
                if n_plots > self.max_cols:
                    n_rows = math.ceil(n_plots / self.max_cols)
                else:
                    n_rows = 1
                    self.max_cols = n_plots

                self.fig, self.axs = plt.subplots(n_rows, self.max_cols,
                                        figsize=(7.6 * self.max_cols, 6 * n_rows),
                                        dpi=self.dpi_display)
                self.fig.tight_layout() # helps to equalize size of subplots. Without the subplots change parameters during plotting which results in differently sized spots.

                if n_plots > 1:
                    self.axs = self.axs.ravel()
                else:
                    self.axs = [self.axs]

                remove_empty_subplots(
                    n_plots=n_plots,
                    nrows=n_rows,
                    ncols=self.max_cols,
                    axis=self.axs
                    )

        else:
            n_plots = len(self.keys)
            if self.max_cols is None:
                self.max_cols = n_plots
                n_rows = 1
            else:
                if n_plots > self.max_cols:
                    n_rows = math.ceil(n_plots / self.max_cols)
                else:
                    n_rows = 1
                    self.max_cols = n_plots

            self.fig, self.axs = plt.subplots(n_rows, self.max_cols,
                                              figsize=(8 * self.max_cols, 6 * n_rows),
                                              dpi=self.dpi_display)

            if n_plots > 1:
                self.axs = self.axs.ravel()
            else:
                self.axs = np.array([self.axs])

            # remove axes from empty plots
            remove_empty_subplots(
                axes=self.axs,
                nplots=n_plots,
                nrows=n_rows,
                ncols=self.max_cols,
                )

        if self.header is not None:
            plt.suptitle(self.header, fontsize=18, x=0.5, y=0.98)

    def plot_to_subplots(self):
        i = 0
        for idx in range(self.n_data):
            # extract this group
            # ad, mask = _extract_groups(
            #     self.adata, self.groupby, group,
            #     extract_uns=False, return_mask=True)
            # get image data for this group

            # extract the InSituData
            try:
                xd = self.data.data[idx]
            except AttributeError:
                xd = self.data
            celldata = _get_cell_layer(cells=xd.cells, cells_layer=self.cells_layer)
            ad = celldata.matrix
            name = xd.sample_id

            if self.image_key is not None:
                imagedata = xd.images
            else:
                imagedata = None
            # ImgD = _ImageData(
            #         adata=ad,
            #         image_key=self.image_key,
            #         group=group,
            #         lowres=self.lowres,
            #         histogram_setting=self.histogram_setting,
            #         )

            for col, key in enumerate(self.keys):
                # create color dictionary if key is categorical

                #color_dict = create_color_dict(ad, key, self.palette)

                if self.crange is None:
                    if key not in self.normalize_crange_not_for:
                        _crange = get_crange(ad, key=key, use_raw=self.raw, layer=self.layer, ctype=self.crange_type)
                    else:
                        _crange = None
                else:
                    _crange = self.crange

                # get axis to plot
                if self.ax is None:
                    if len(self.axs.shape) == 2:
                        ax = self.axs[idx, col]
                    elif len(self.axs.shape) == 1:
                        if self.multikeys:
                            ax = self.axs[col]
                        else:
                            ax = self.axs[i]
                    else:
                        raise ValueError("`len(self.axs.shape)` has wrong shape {}. Requires 1 or 2.".format(len(self.axs.shape)))
                else:
                    ax = self.ax

                # counter for axis
                i+=1

                # get data
                ConfigData = _ConfigSpatialPlot(
                    adata=ad,
                    key=key,
                    ImageDataObject=imagedata,
                    image_key=self.image_key,
                    image_pyramid_level=self.image_pyramid_level,
                    raw=self.raw,
                    layer=self.layer,
                    obsm_key=self.obsm_key,
                    origin_zero=self.origin_zero,
                    xlim=self.xlim,
                    ylim=self.ylim,
                    spot_size=self.spot_size,
                    margin=self.margin
                )

                if ConfigData.exists:
                    # set axis
                    ax.set_xlim(ConfigData.xlim[0], ConfigData.xlim[1])
                    ax.set_ylim(ConfigData.ylim[0], ConfigData.ylim[1])

                    # if imagedata.image_metadata is None or self.plot_pixel:
                    #     ax.set_xlabel('pixels', fontsize=14)
                    #     ax.set_ylabel('pixels', fontsize=14)
                    # else:
                    ax.set_xlabel('µm', fontsize=14)
                    ax.set_ylabel('µm', fontsize=14)

                    ax.invert_yaxis()
                    ax.grid(False)
                    ax.set_aspect(1)
                    ax.set_facecolor(self.background_color)
                    ax.tick_params(labelsize=12)

                    if self.multigroups and not self.multikeys:
                        ax.set_title(name + "\n" + ConfigData.key, fontsize=14, fontweight='bold', rotation=90)
                    else:
                        # set titles
                        ax.set_title(ConfigData.key, fontsize=14, fontweight='bold')

                        if col == 0:
                            ax.annotate(name,
                                        xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                                        xycoords=ax.yaxis.label, textcoords='offset points',
                                        size=14, rotation=90,
                                        ha='right', va='center', weight='bold')

                    if ConfigData.categorical:
                        color_dict = create_cmap_mapping(data=ad.obs[key])
                    else:
                        color_dict = self.palette

                    # if color_dict is None:
                    #     color_dict = self.palette

                    # plot single spatial plot in given axis
                    self.single_spatial(
                        ConfigData=ConfigData,
                        #ImgData=imagedata,
                        #size=size,
                        axis=ax,
                        color_dict=color_dict,
                        crange=_crange
                        )
                else:
                    print("Key '{}' not found.".format(key), flush=True)
                    ax.set_axis_off()

                # free RAM
                del ConfigData
                gc.collect()

            # free RAM
            del imagedata
            gc.collect()

    def single_spatial(
        self,
        ConfigData: Type[_ConfigSpatialPlot],
        #ImgData: Type[_ImageData],
        #size: float,
        axis: plt.Axes,
        color_dict: Dict,
        crange: Optional[Tuple[float, float]]
        ):

        # calculate marker size
        pixels_per_unit = axis.transData.transform(
            [(0, 1), (1, 0)]) - axis.transData.transform((0, 0))
        # x_ppu = pixels_per_unit[1, 0]
        y_ppu = pixels_per_unit[0, 1]
        pxs = y_ppu * ConfigData.spot_size
        size = (72. / self.fig.dpi * pxs)**2

        # if self.plot_pixel:
        #     # plot image data
        #     if ConfigData.image is not None:
        #         axis.imshow(ConfigData.image, origin='upper', cmap='gray')

        #     # plot transcriptomic data
        #     s = axis.scatter(
        #         ConfigData.x_pixelcoord * ImgData.scale_factor,
        #         ConfigData.y_pixelcoord * ImgData.scale_factor,
        #         c=ConfigData.color, marker=self.spot_type,
        #         s=ConfigData.spot_size, alpha=self.alpha,
        #         linewidths=0,
        #         cmap=self.cmap
        #         )
        # else:
        # plot image data
        if ConfigData.image is not None:
            axis.imshow(
                ConfigData.image,
                extent=(
                    -0.5 - ConfigData.x_offset,
                    ConfigData.image.shape[1] * ConfigData.pixel_size - 0.5 - ConfigData.x_offset,
                    ConfigData.image.shape[0] * ConfigData.pixel_size - 0.5 - ConfigData.y_offset,
                    # ConfigData.image.shape[1] / ConfigData.pixel_per_um / ConfigData.scale_factor - 0.5 - ConfigData.x_offset,
                    # ConfigData.image.shape[0] / ConfigData.pixel_per_um / ConfigData.scale_factor - 0.5 - ConfigData.y_offset,
                    -0.5 - ConfigData.y_offset
                    ),
                origin='upper', cmap='gray')
        # plot transcriptomic data
        if not ConfigData.categorical:
            s = axis.scatter(
                ConfigData.x_coords, ConfigData.y_coords, c=ConfigData.color,
                marker=self.spot_type,
                #s=ConfigData.spot_size,
                s=size,
                alpha=self.alpha,
                linewidths=0,
                cmap=self.cmap,
                norm=self.normalize
                )
        else:
            sns.scatterplot(
                data=ConfigData.data,
                x='x_coord', y='y_coord',
                hue=ConfigData.key,
                marker=self.spot_type,
                #s=ConfigData.spot_size,
                s=size,
                #edgecolor="none",
                linewidth=0,
                palette=color_dict, alpha=self.alpha,
                ax=axis
                )
        # plot legend
        if ConfigData.categorical:
            # divide axis to fit legend
            divider = make_axes_locatable(axis)
            lax = divider.append_axes("right", size="4%", pad=0)

            # Get handles and labels from the axis
            handles, labels = axis.get_legend_handles_labels()

            # Create a legend manually
            legend = lax.legend(handles, labels, loc='center left', ncol=2, frameon=False)

            # Remove the axis ticks and labels
            lax.set_xticks([])
            lax.set_yticks([])
            lax.axis('off')

            # Adjust the size of the legend markers
            for handle in legend.legend_handles:
                handle.set_markersize(12)  # Adjust the size as needed
                handle.set_markeredgecolor('black')  # Set the edge color to black
                handle.set_markeredgewidth(1.5)  # Set the edge width

            # Remove the legend from the main axis
            axis.legend().remove()
        else:
            if self.colorbar:
                # divide axis to fit colorbar
                divider = make_axes_locatable(axis)
                cax = divider.append_axes("right", size="4%", pad=0.1)
                clb = self.fig.colorbar(s, cax=cax)

                # set colorbar
                clb.ax.tick_params(labelsize=14)

                if self.clb_title is not None:
                    clb.ax.set_ylabel(self.clb_title,
                                    rotation=270,
                                    fontdict={"fontsize": 14},
                                    labelpad=20)

                if crange is not None:
                    clb.mappable.set_clim(crange[0], crange[1])
                else:
                    if self.crange_type == 'percentile':
                        clb.mappable.set_clim(0, np.percentile(ConfigData.color, 99))
        # # plot legend
        # if ConfigData.categorical:
        #     legend = axis.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

        #     # Adjust the size of the legend markers
        #     for handle in legend.legend_handles:
        #         handle.set_markersize(12)  # Adjust the size as needed
        #         handle.set_markeredgecolor('black')  # Set the edge color to black
        #         handle.set_markeredgewidth(1.5)  # Set the edge width
        # else:
        #     if self.colorbar:
        #         # divide axis to fit colorbar
        #         divider = make_axes_locatable(axis)
        #         cax = divider.append_axes("right", size="4%", pad=0.1)
        #         clb = self.fig.colorbar(s, cax=cax)

        #         # set colorbar
        #         clb.ax.tick_params(labelsize=14)

        #         if self.clb_title is not None:
        #             clb.ax.set_ylabel(self.clb_title,
        #                               rotation=270,
        #                               fontdict={"fontsize": 14},
        #                               labelpad=20)

        #         if crange is not None:
        #                 clb.mappable.set_clim(crange[0], crange[1])
        #         else:
        #             if self.crange_type == 'percentile':
        #                 clb.mappable.set_clim(0, np.percentile(ConfigData.color, 99))

