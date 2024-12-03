import os
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from napari.viewer import Viewer

import insitupy._core.config as config
from insitupy.io.plots import save_and_show_figure
from insitupy.plotting._colors import _data_to_rgba


def plot_colorlegend(
    viewer: Viewer,
    layer_name: Optional[str] = None,
    savepath: Union[str, os.PathLike, Path] = None,
    save_only: bool = False,
    dpi_save: int = 300,
    ):
    # automatically get layer
    if layer_name is None:
        candidate_layers = [l for l in viewer.layers if l.name.startswith(f"{config.current_data_name}")]
        try:
            layer_name = candidate_layers[0].name
        except IndexError:
            raise ValueError("No layer with cellular transcriptomic data found. First add a layer using the 'Show Data' widget.")

    # extract layer
    layer = viewer.layers[layer_name]

    # get values
    values = layer.properties["value"]

    # create color mapping
    rgba_list, mapping, cmap = _data_to_rgba(values)

    if isinstance(mapping, dict):
        # categorical colorbar
        # create a figure for the colorbar
        fig, ax = plt.subplots(
            #figsize=(5, 3)
            )
        fig.subplots_adjust(bottom=0.5)

        circles = [Line2D([0], [0],
                            marker='o', color='w', label=label,
                            markerfacecolor=color, markeredgecolor='k', markersize=15) for label, color in mapping.items()]

        ax.legend(handles=circles, loc="center", labelspacing=1, borderpad=0.5)
        ax.set_axis_off()

    else:
        # continuous colorlegend
        # create a figure for the colorbar
        fig, ax = plt.subplots(
            figsize=(6, 1)
            )
        fig.subplots_adjust(bottom=0.5)

        # Add the colorbar to the figure
        cbar = fig.colorbar(mapping, orientation='horizontal', cax=ax)
        cbar.ax.set_title(layer_name)

    save_and_show_figure(savepath=savepath, fig=fig, save_only=save_only, dpi_save=dpi_save, tight=False)
    plt.show()


