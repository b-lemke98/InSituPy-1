import os
import textwrap
from pathlib import Path
from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from napari.viewer import Viewer

import insitupy._core.config as config
from insitupy._constants import DEFAULT_CATEGORICAL_CMAP
from insitupy._core._checks import _check_assignment
from insitupy.io.plots import save_and_show_figure
from insitupy.plotting._colors import _data_to_rgba
from insitupy.utils.utils import get_nrows_maxcols


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


def plot_cellular_composition(
    data,
    cell_type_col: str,
    key: str,
    modality: Literal["regions", "annotations"] = "regions",
    plot_type: Literal["pie", "bar"] = "pie",
    force_assignment: bool = False,
    max_cols: int = 4,
    savepath: Union[str, os.PathLike, Path] = None,
    show_labels: bool = False,
    adjust_labels: bool = False,
    label_threshold: float = 2.,
    return_data: bool = False,
    save_only: bool = False,
    dpi_save: int = 300,
    ):

    """
    Plots the composition of cell types for specified regions or annotations.

    This function generates pie charts or a single stacked bar plot to visualize the proportions of different cell types
    within specified regions or annotations. It can optionally save the plot to a file and
    return the composition data.

    Args:
        data: The dataset containing cell information.
        cell_type_col (str): The column name in `adata.obs` that contains cell type information.
        key (str): The key to access the specific annotation or region in `adata.obsm`.
        modality (Literal["regions", "annotations"], optional): The modality to use, either "regions" or "annotations". Default is "regions".
        plot_type (Literal["pie", "bar"], optional): The type of plot to generate, either "pie" or "bar". Default is "pie".
        force_assignment (bool, optional): If True, forces reassignment of cells to the requested annotation. Default is False.
        max_cols (int, optional): Maximum number of columns for subplots. Defaults to 4.
        savepath (Union[str, os.PathLike, Path], optional): The path to save the plot. If None, the plot is not saved. Default is None.
        show_labels (bool, optional): If True, displays percentage labels on the pie charts. Default is False.
        adjust_labels (bool, optional): If True, adjusts the labels to avoid overlap. Default is False.
        label_threshold (float, optional): The threshold percentage above which labels are displayed. Default is 2.0.
        return_data (bool, optional): If True, returns the composition data as a DataFrame. Default is False.
        save_only (bool, optional): If True, only saves the plot without displaying it. Default is False.
        dpi_save (int, optional): The resolution in dots per inch for the saved plot. Default is 300.

    Returns:
        pd.DataFrame: A DataFrame containing the composition of cell types if `return_data` is True.

    Raises:
        ValueError: If the specified key or modality is not found in the data.

    Example:
        >>> compositions = plot_cell_composition(data, cell_type_col="cell_type", key="region_1", plot_type="bar", return_data=True)
        >>> print(compositions)
    """
    if adjust_labels:
        try:
            from adjustText import adjust_text
        except ImportError:
            raise ImportError("The 'adjustText' module is required for label adjustment. Please install it with `pip install adjusttext` or select adjust_labels=False.")

    # check whether the cells were already assigned to the requested annotation
    _check_assignment(data=data, key=key, force_assignment=force_assignment, modality=modality)

    # retrieve data
    adata = data.cells.matrix
    assignment_series = adata.obsm[modality][key]
    cats = sorted([elem for elem in assignment_series.unique() if (elem != "unassigned") & ("&" not in elem)])

    # calculate compositions
    compositions = {}
    for cat in cats:
        idx = assignment_series[assignment_series.str.contains(cat)].index
        compositions[cat] = adata.obs[cell_type_col].loc[idx].value_counts(normalize=True) * 100 # calculate percentage
    compositions = pd.DataFrame(compositions)

    # Define a function to display percentages above the threshold
    def autopct_func(pct):
        return ('%1.1f%%' % pct) if pct > label_threshold else ''

    if plot_type == "pie":
        # Plot pie charts for each area
        n_plots, nrows, ncols = get_nrows_maxcols(len(cats), max_cols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(6*ncols,6*nrows))

        if n_plots > 1:
            axs = axs.ravel()
        else:
            axs = [axs]

        for i, area in enumerate(compositions.columns):
            if show_labels:
                wedges, texts, autotexts = axs[i].pie(compositions[area],
                                                    autopct=autopct_func, pctdistance=1.15,
                                                    colors=DEFAULT_CATEGORICAL_CMAP.colors
                                                    )
            else:
                wedges, texts = axs[i].pie(compositions[area],
                                                    colors=DEFAULT_CATEGORICAL_CMAP.colors
                                                    )

            title_str = textwrap.fill(f'Proportions of Cell Types in {area}', width=20)
            axs[i].set_title(title_str)

            if adjust_labels:
                # Adjust text to avoid overlap
                adjust_text(texts + autotexts, ax=axs[i], arrowprops=dict(arrowstyle="->", color='k', lw=0.5))

        # Add a legend
        fig.legend(wedges, compositions.index, loc='center left', bbox_to_anchor=(0.92, 0.5))

    elif plot_type == "bar":
        # Plot a single stacked bar plot
        compositions.T.plot(kind='bar', stacked=True, figsize=(1*len(cats), 6), width=0.7, color=DEFAULT_CATEGORICAL_CMAP.colors)
        plt.title('Proportions of Cell Types by Area')
        plt.ylabel('%')
        plt.xlabel('Area')
        plt.legend(title='Cell Types', bbox_to_anchor=(1.05, 1), loc='upper left')

    save_and_show_figure(savepath=savepath, fig=plt.gcf(), save_only=save_only, dpi_save=dpi_save, tight=False)

    plt.tight_layout()
    plt.show()

    if return_data:
        return compositions
