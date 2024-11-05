import os
from numbers import Number
from typing import List, Literal, Optional, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from insitupy._constants import DEFAULT_CATEGORICAL_CMAP
from insitupy._core._checks import check_raw, has_valid_labels
from insitupy.io.plots import save_and_show_figure
from insitupy.utils._regression import smooth_fit
from insitupy.utils.utils import convert_to_list, get_nrows_maxcols


def expr_along_obs_val(
    adata: AnnData,
    keys: str,
    obs_val: Union[str, Tuple[str, str]],
    groupby: Optional[str] = None,
    splitby: str = None,
    hue: str = None,
    method: Literal["lowess", "loess"] = 'loess',
    stderr: bool = False,
    loess_bootstrap: bool = True,
    n_bootstraps_iterations: int = 100,
    xmin=None,
    xmax=None,
    cmap="tab10",
    linewidth=8,
    extra_cats=None,
    normalize=False,
    nsteps=100,
    show_progress=False,
    use_raw=False,
    max_cols=4,
    xlabel=None,
    ylabel=None,
    vline=None,
    hline=None,
    vlinewidth=4,
    custom_titles=None,
    legend_fontsize=24,
    plot_legend=True,
    xlabel_fontsize=28,
    ylabel_fontsize=28,
    title_fontsize=20,
    tick_fontsize=24,
    figsize=(8,6),
    savepath: Optional[os.PathLike] = None,
    save_only: bool = False,
    show: bool = True,
    return_data: bool = False,
    fig: Optional[Figure] = None,
    axis: Optional[Axes] = None,
    dpi_save: int = 300,
    smooth=True,
    **kwargs
    ):
    """
    Plot gene expression values along a specified observation category.

    Args:
        adata (AnnData): Annotated data matrix.
        keys (str): Keys for the gene expression values to be plotted.
        obs_val (Union[str, Tuple[str, str]]): Observation category to be plotted on the x-axis.
            Can be a string representing a column in `adata.obs` or a tuple (obsm_key, obsm_col)
            where `obsm_key` is a key in `adata.obsm` and `obsm_col` is a column in the corresponding DataFrame.
        groupby (Optional[str]): Observation category to group by.
        splitby (str, optional): Observation category to split by.
        hue (str, optional): Observation category to color by.
        method (Literal["lowess", "loess"], optional): Smoothing method to use. Defaults to 'loess'.
        stderr (bool, optional): Whether to plot standard error. Defaults to False.
        loess_bootstrap (bool, optional): Whether to use bootstrap for loess smoothing. Defaults to True.
        n_bootstraps_iterations (int, optional): Number of bootstrap iterations for loess smoothing. Defaults to 100.
        xmin (optional): Minimum x value for plotting.
        xmax (optional): Maximum x value for plotting.
        cmap (str, optional): Colormap to use for plotting. Defaults to "tab10".
        linewidth (int, optional): Line width for plotting. Defaults to 8.
        extra_cats (optional): Additional observation categories to include in the plot.
        normalize (bool, optional): Whether to normalize the expression values. Defaults to False.
        nsteps (int, optional): Number of steps for smoothing. Defaults to 100.
        show_progress (bool, optional): Whether to show progress bar. Defaults to False.
        use_raw (bool, optional): Whether to use raw data. Defaults to False.
        max_cols (int, optional): Maximum number of columns for subplots. Defaults to 4.
        xlabel (optional): Label for the x-axis.
        ylabel (optional): Label for the y-axis.
        vline (optional): Vertical lines to add to the plot.
        hline (optional): Horizontal lines to add to the plot.
        vlinewidth (int, optional): Line width for vertical lines. Defaults to 4.
        custom_titles (optional): Custom titles for the plots.
        legend_fontsize (int, optional): Font size for the legend. Defaults to 24.
        plot_legend (bool, optional): Whether to plot the legend. Defaults to True.
        xlabel_fontsize (int, optional): Font size for the x-axis label. Defaults to 28.
        ylabel_fontsize (int, optional): Font size for the y-axis label. Defaults to 28.
        title_fontsize (int, optional): Font size for the plot titles. Defaults to 20.
        tick_fontsize (int, optional): Font size for the axis ticks. Defaults to 24.
        figsize (tuple, optional): Figure size. Defaults to (8, 6).
        savepath (optional): Path to save the plot.
        save_only (bool, optional): Whether to only save the plot without showing. Defaults to False.
        show (bool, optional): Whether to show the plot. Defaults to True.
        axis (optional): Axis to plot on.
        return_data (bool, optional): Whether to return the data instead of plotting. Defaults to False.
        fig (optional): Figure to plot on.
        dpi_save (int, optional): DPI for saving the plot. Defaults to 300.
        smooth (bool, optional): Whether to apply smoothing. Defaults to True.
        **kwargs: Additional arguments for smoothing.

    Returns:
        Union[DataFrame, Tuple[Figure, Axes]]:
            If return_data is True, returns a DataFrame with the smoothed data.
            Otherwise, returns the figure and axes of the plot.
    """

    # check type of input
    if isinstance(keys, dict):
        if custom_titles is not None:
            print("Attention: `custom_titles` was not None and `keys` was dictionary. Titles were retrieved from dictionary.")
        custom_titles = list(keys.keys())
        keys = list(keys.values())

    # make inputs to lists
    keys = [keys] if isinstance(keys, str) else list(keys)


    adata_obs = adata.obs.copy()
    if isinstance(obs_val, tuple):
        print("Retrieve `obs_val` from .obsm.")
        obsm_key = obs_val[0]
        obsm_col = obs_val[1]
        obs_val = f"distance_from_{obsm_col}"
        adata_obs[obs_val] = adata.obsm[obsm_key][obsm_col]

    # remove NaNs `obs_val` column
    not_na_and_not_zero_mask = adata_obs[obs_val].notna() & adata_obs[obs_val] > 0
    adata_obs = adata_obs[not_na_and_not_zero_mask]

    # check whether to plot raw data
    X, var, var_names = check_raw(adata, use_raw=use_raw)

    # remove rows from X which were NaN above
    X = X[not_na_and_not_zero_mask]

    if hue is not None:
        hue_cats = list(adata_obs[hue].unique())
        cmap_colors = plt.get_cmap(cmap)
        color_dict = {a: cmap_colors(i) for i, a in enumerate(hue_cats)}

        if extra_cats is None:
            extra_cats = [hue]
        else:
            extra_cats.append(hue)

    #if show:
    if not return_data:
        # prepare plotting
        if axis is None:
            n_plots, n_rows, max_cols = get_nrows_maxcols(keys, max_cols)
            fig, axs = plt.subplots(n_rows,max_cols, figsize=(figsize[0]*max_cols, figsize[1]*n_rows))

        else:
            axs = axis
            #fig = None
            n_plots = 1
            show = False # otherwise plotting into given axes wouldn't work

        if n_plots > 1:
            axs = axs.ravel()
        else:
            axs = [axs]

    data_collection = {}
    for i, key in (enumerate(tqdm(keys)) if show_progress else enumerate(keys)):
        # check if the keys are also grouped
        keys_grouped = isinstance(key, list)

        if groupby is not None:
            # select data per group
            groups = adata_obs[groupby].unique()
        else:
            groups = [None]

        added_to_legend = []

        group_collection = {}
        for group in groups:
            #partial = extract_groups(adata, groupby=groupby, groups=group)

            if group is not None:
                group_mask = adata_obs[groupby] == group
                group_obs = adata_obs.loc[group_mask, :].copy()
            else:
                group_mask = [True] * len(adata_obs)
                group_obs = adata_obs

            if hue is not None:
                _hue = adata_obs.loc[group_mask, hue][0]

            # hue_data = adata_obs.loc[group_mask, hue].copy()
            # print(hue_data)

            # select only group values from matrix
            group_X = X[group_mask, :]

            if splitby is None:
                # select x value
                x = group_obs.loc[:, obs_val].values
                # if xmin is None:
                #     xmin = x[x>0].min()
                #     print(xmin, flush=True)

                if keys_grouped:
                    # extract expression values of all keys in the group
                    idx = var.index.get_indexer(key)
                    dd = pd.DataFrame(group_X[:, idx], index=x)

                    if normalize:
                        #dd = dd.apply(minmax_scale, axis=0)
                        dd = dd.apply(zscore, axis=0)

                    dd = dd.reset_index().melt(id_vars="index") # reshape to get long list of x values
                    x = dd["index"].values
                    y = dd["value"].values

                elif key in var_names:
                    # extract expression values as y
                    idx = var.index.get_loc(key)
                    y = group_X[:, idx].copy()

                    if normalize:
                        #y = minmax_scale(y)
                        y = zscore(y)

                elif key in group_obs.columns:
                    y = group_obs.loc[:, key].values.copy()
                else:
                    print("Key '{}' not found.".format(key))
                    break

                if smooth:
                    # do smooth fitting
                    df = smooth_fit(x, y,
                                    xmax=xmin, xmin=xmax,
                                    nsteps=nsteps, method=method,
                                    stderr=stderr, loess_bootstrap=loess_bootstrap,
                                    K=n_bootstraps_iterations,
                                    **kwargs)
                else:
                    # set up dataframe without smooth fitting
                    df = pd.DataFrame({"x": x, "y_pred": y})

                if extra_cats is not None:
                    df = df.join(adata_obs.loc[group_mask, extra_cats].reset_index(drop=True))

            else:
                splits = group_obs[splitby].unique()
                df_collection = {}

                # get min and max values for x values
                x = group_obs[obs_val].values
                xmin = x.min()
                xmax = x.max()

                for split in splits:
                    # extract x values
                    split_mask = group_obs[splitby] == split
                    x = group_obs.loc[split_mask, obs_val].values

                    # extract expression values as y
                    idx = var.index.get_loc(key)
                    y = group_X[split_mask, idx].copy()

                    # do smooth fitting
                    if smooth:
                        df_split = smooth_fit(x, y,
                                xmax=xmin, xmin=xmax,
                                nsteps=nsteps, method=method, stderr=stderr, **kwargs)
                    else:
                        # set up dataframe without smooth fitting
                        df_split = pd.DataFrame({"x": x, "y_pred": y})

                    # collect data
                    df_collection[split] = df_split

                df_collection = pd.concat(df_collection)

                # calculate mean and std
                df = df_collection[['x', 'y_pred']].groupby('x').mean()
                df['std'] = df_collection[['x', 'y_pred']].groupby('x').std()
                df['conf_lower'] = [a-b for a,b in zip(df['y_pred'], df['std'])]
                df['conf_upper'] = [a+b for a,b in zip(df['y_pred'], df['std'])]
                df.reset_index(inplace=True)

            # remove NaNs
            df = df.dropna(how="all", axis=1)

            if return_data:
                group_collection[group] = df
            else:
                # sort by x-value
                df.sort_values('x', inplace=True)

                # plotting
                cols = df.columns
                if 'conf_lower' in cols and 'conf_upper' in cols:
                    axs[i].fill_between(df['x'],
                                    df['conf_lower'],
                                    df['conf_upper'],
                                    alpha = 0.2,
                                    color = 'grey')

                # determine label variable
                if hue is not None:
                    label = _hue if _hue not in added_to_legend else ""
                    color = color_dict[_hue]
                else:
                    label = group
                    color = None

                axs[i].plot(df['x'],
                    df['y_pred'],
                    label=label,
                    color=color,
                    linewidth=linewidth)

                if hue is not None and _hue not in added_to_legend:
                    added_to_legend.append(_hue)

        # optionally add vertical or horizontal lines to plot
        if vline is not None:
            if isinstance(vline, dict):
                linecolors = list(vline.keys())
                vline = list(vline.values())
            else:
                vline = [vline] if isinstance(vline, int) or isinstance(vline, float) else list(vline)
                linecolors = ['k'] * len(vline)

            for c, v in zip(linecolors, vline):
                axs[i].axvline(x=v, ymin=0, ymax=1, c=c, linewidth=vlinewidth, linestyle='dashed')

        if hline is not None:
            if isinstance(hline, dict):
                linecolors = list(hline.keys())
                hline = list(hline.values())
            else:
                hline = [hline] if isinstance(hline, int) or isinstance(hline, float) else list(hline)
                linecolors = ['k'] * len(hline)

            for c, h in zip(linecolors, hline):
                axs[i].axhline(y=h, xmin=0, xmax=1, c=c, linewidth=4, linestyle='dashed')

        if not return_data:
            if xlabel is None:
                xlabel = obs_val
            if ylabel is None:
                ylabel = "Gene expression"

            axs[i].set_xlabel(xlabel, fontsize=xlabel_fontsize)
            axs[i].set_ylabel(ylabel, fontsize=ylabel_fontsize)
            axs[i].tick_params(axis='both', which='major', labelsize=tick_fontsize)
            #axs[i].set_xlim(0, 1)
            #axs[i].xaxis.set_major_locator(ticker.FixedLocator([0.1, 0.9]))

            if custom_titles is None:
                axs[i].set_title(key, fontsize=title_fontsize)
            else:
                assert len(custom_titles) == len(keys), "List of title values has not the same length as list of keys."
                axs[i].set_title(str(custom_titles[i]), fontsize=title_fontsize)

            if plot_legend:
                if has_valid_labels(axs[i]):
                    axs[i].legend(fontsize=legend_fontsize,
                    loc='best'
                    )
            else:
                # first check if there are valid labels in the axis to circumvent warning
                if has_valid_labels(axs[i]):
                    axs[i].legend().remove()

        if return_data:
            if len(group_collection) > 0:
                # collect data
                group_collection = pd.concat(group_collection)
                data_collection[key] = group_collection
            else:
                pass

    if return_data:
        # close plot
        plt.close()

        # return data
        data_collection = pd.concat(data_collection)
        data_collection.index.names = ['key', groupby, None]
        return data_collection

    else:
        if n_plots > 1:

            # check if there are empty plots remaining
            while i < n_rows * max_cols - 1:
                i+=1
                # remove empty plots
                axs[i].set_axis_off()
        if show:
            #fig.tight_layout()
            save_and_show_figure(savepath=savepath, fig=fig, save_only=save_only, dpi_save=dpi_save, tight=True)
        else:
            return fig, axs

def cell_expression_along_axis(
    adata,
    obs_val,
    genes: List[str],
    cell_type_column,
    cell_type,
    xlim: Tuple[Union[int, float], Union[int, float]] = (0, np.inf),
    min_expression: Union[int, float] = 0,
    xlabel: Optional[str] = None,
    fit_reg: bool = False,
    lowess: bool = False,
    robust: bool = False,
    fig_height: Number = 4,
    fig_marginal_ratio: Number = 0.2,
    scatter_size: Number = 1
):

    """
    Plot gene expression along a specified axis for a given cell type.

    Args:
        adata: AnnData object containing the single-cell data.
        obs_val: Observation value to plot along the x-axis.
        genes (List[str]): List of genes to plot.
        cell_type_column: Column name in `adata.obs` that contains cell type information.
        cell_type: Specific cell type to filter the data.
        xlim (Tuple[Union[int, float], Union[int, float]], optional): Limits for the x-axis. Defaults to (0, np.inf).
        min_expression (Union[int, float], optional): Minimum expression level to include in the plot. Defaults to 0.
        xlabel (Optional[str], optional): Label for the x-axis. Defaults to None.
        fit_reg (bool, optional): Whether to fit a regression line. Defaults to False.
        lowess (bool, optional): Whether to use LOWESS for regression. Defaults to False.
        robust (bool, optional): Whether to use a robust regression. Defaults to False.
        fig_height (Number, optional): Height of the figure. Defaults to 4.
        fig_marginal_ratio (Number, optional): Ratio of the marginal plot height to the main plot height. Defaults to 0.2.
        scatter_size (Number, optional): Size of the scatter plot points. Defaults to 1.

    Returns:
        None: Displays the plot.
    """
    # make sure genes is a list
    genes = convert_to_list(genes)

    # select the data for plotting
    data_for_one_celltype = _select_data(
        adata=adata,
        obs_val=obs_val,
        cell_type_column=cell_type_column,
        cell_type=cell_type,
        genes=genes,
        xlim=xlim,
    )

    # data_dict = {gene: _select_data(
    #     adata=adata,
    #     obs_val=obs_val,
    #     cell_type_column=cell_type_column,
    #     cell_type=cell_type,
    #     gene=gene,
    #     xlim=xlim,
    # )
    #  for gene in genes
    #  }

    # Prepare a figure with subplots
    num_genes = len(genes)
    marg_height = fig_height * fig_marginal_ratio
    fig, axes = plt.subplots(num_genes + 1, 2,
                             figsize=(fig_height + marg_height, fig_height * (num_genes) + marg_height),
                             #sharex=True, sharey=True,
                             sharey='row', sharex='col',
                             height_ratios=[marg_height] + [fig_height]*num_genes,
                             width_ratios=[fig_height, marg_height]
                             )

    # Histogram for the x-axis density
    #data_for_axis_histogram = data_dict[list(data_dict.keys())[0]]
    sns.kdeplot(data=data_for_one_celltype, x="axis", ax=axes[0, 0], color='darkgray', fill=True)

    # remove values axis from histogram
    axes[0, 0].get_yaxis().set_visible(False)
    axes[0, 0].spines['left'].set_visible(False)

    for i, gene in enumerate(genes):
    #for i, (gene, data_of_one_celltype) in enumerate(data_dict.items()):

        data_for_one_gene = data_for_one_celltype[["axis", gene]].copy()

        # Apply limits
        data_filtered = data_for_one_gene[data_for_one_gene[gene] >= min_expression]

        # KDE plot
        sns.kdeplot(data=data_filtered, x="axis", y=gene, ax=axes[i + 1, 0], fill=True, cmap="Reds", levels=8)

        # Scatter plot
        sns.regplot(data=data_filtered,
                    x="axis", y=gene, ax=axes[i + 1, 0],
                    color="k", #s=8
                    scatter_kws={"s": scatter_size},
                    fit_reg=fit_reg,
                    lowess=lowess,
                    robust=robust,
                    line_kws={"color": "orange"}
                    )

        # Histogram for the gene expression
        sns.kdeplot(
            data=data_filtered, y=gene, ax=axes[i + 1, 1], color='darkgray', fill=True
        )

        # remove values axis from histogram
        axes[i + 1, 1].get_xaxis().set_visible(False)
        axes[i + 1, 1].spines['bottom'].set_visible(False)

        # Set labels
        axes[i + 1, 0].set_ylabel(f"{gene} in '{cell_type}'")

    # Set common x-label
    if xlabel is None:
        axes[-1, 0].set_xlabel("_".join(convert_to_list(obs_val)))
    else:
        axes[-1, 0].set_xlabel(xlabel)

    axes[0, 1].remove()

    plt.tight_layout()
    plt.show()

def _select_data(
    adata,
    obs_val,
    genes: List[str],
    cell_type_column,
    cell_type,
    xlim: Tuple[Union[int, float], Union[int, float]] = (0, np.inf),
    min_expression: Number = None,
    sort: bool = True,
    minmax_scale: bool = True,
    verbose: bool = True
):
    # make sure genes is a list
    genes = convert_to_list(genes)

    # Check type of obs_val
    adata_obs = adata.obs.copy()
    if isinstance(obs_val, tuple):
        print("Retrieve `obs_val` from .obsm.") if verbose else None
        obsm_key = obs_val[0]
        obsm_col = obs_val[1]
        #obs_val = f"distance_from_{obsm_col}"
        adata_obs["axis"] = adata.obsm[obsm_key][obsm_col]

    # Get data for plotting
    data = adata_obs[["axis", cell_type_column]].dropna()

    # Filter data for the specified cell type
    selected_data = data[data[cell_type_column] == cell_type].copy()

    # Apply limits
    selected_data = selected_data[
        (selected_data["axis"] >= xlim[0]) &
        (selected_data["axis"] <= xlim[1])
        ]

    for i, gene in enumerate(genes):
        # Add gene expression information
        gene_loc = adata.var_names.get_loc(gene)
        expr = adata.X[:, gene_loc]
        expr = pd.Series(expr.toarray().flatten(), index=adata.obs_names)

        if min_expression is not None:
            # mask values below the threshold with NaN
            expr = expr.mask(expr < min_expression)

        # add gene expression to dataframe
        selected_data[gene] = expr

        # if min_expression is not None:
        #     # Apply limits
        #     #data_for_one_celltype = data_for_one_celltype[data_for_one_celltype[gene] >= min_expression]

    # drop the cell type column
    #selected_data = selected_data.drop(cell_type_column, axis=1)

    # add axis column to index
    selected_data = selected_data.set_index(["axis", cell_type_column], append=True)
    selected_data.index.names = ["cell_id", "axis", cell_type_column]

    if sort:
        #selected_data = selected_data.sort_values("axis")
        selected_data = selected_data.sort_index(level='axis', ascending=True)

    if minmax_scale:
        scaler = MinMaxScaler()
        selected_data = pd.DataFrame(scaler.fit_transform(selected_data),
                    index=selected_data.index, columns=selected_data.columns
                    )

    return selected_data

# binning function
def _bin_data(data,
              #expr, axis,
              axis_name: str = "axis",
              resolution=5,
              #minmax_scale: bool = True,
              plot: bool = False
              ):

    # make a copy of the data
    data = data.copy()

    # get values
    axis = data.index.get_level_values(axis_name).values

    # calculate number of bins from resolution
    nbins = int((axis.max() - axis.min()) / resolution)
    #nbins = int((data[axis_name].max() - data[axis_name].min()) / resolution)

    # data = pd.DataFrame(
    #     {"axis": axis,
    #      "expr": expr}
    # )

    # get gene names
    genes = [elem for elem in data.columns if elem != axis_name]

    # bin data and calculate mean per bin
    data["bin"] = pd.cut(data[axis_name], bins=nbins)

    binned_mean = data.groupby("bin")[genes].mean()
    #binned_mean = binned_mean.reset_index()

    # remove empty bins
    #binned_mean = binned_mean.dropna()

    # if minmax_scale:
    #     # scale values to 0-1
    #     # scaler = MinMaxScaler()
    #     # binned_mean["minmax"] = scaler.fit_transform(binned_mean[[expr_col]])
    #     scaler = MinMaxScaler()
    #     binned_mean = pd.DataFrame(scaler.fit_transform(binned_mean),
    #                 index=binned_mean.index,
    #                 columns=binned_mean.columns)

    # extract the center of each bin
    #binned_mean["bin_center"] = [elem.mid for elem in binned_mean["bin"]]
    binned_mean.index = [elem.mid for elem in binned_mean.index]

    if plot:
        for gene in genes:
            _bin_qc_plot(
                raw_axis=data[axis_name].values,
                raw_expr=data[gene].values,
                bin_center=binned_mean.index.values,
                expr=binned_mean[gene].values,
                ylabel=gene
            )

    return binned_mean

def _bin_qc_plot(
    raw_axis, raw_expr, bin_center, expr, xlabel='x', ylabel='y'
):
    # bin_center = binned_data.index.values
    # expr = binned_data["expr"].values

    try:
        # perform loess regression for the second half of the plot
        res = smooth_fit(
        xs=bin_center,
        ys=expr,
        loess_bootstrap=False, nsteps=100
        )
    except ValueError as e:
        print(f"A ValueError occurred during loess regression: {e}")
        res = None

    fig, axs = plt.subplots(1,2, figsize=(8,4))
    # Plot the original data
    axs[0].scatter(
        raw_axis, raw_expr, label='Original Data', alpha=0.5, color='k', s=1
        )

    # Plot the binned values
    axs[0].plot(
        bin_center, expr,
        color='firebrick',
        #marker='o',
        linestyle='-', label='Binned Mean')

    # Add labels and legend
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)
    axs[0].legend()

    #axs[1].plot(binned_data["bin_center"], binned_data["minmax"])
    axs[1].scatter(
        x=bin_center, y=expr,s=1, color="k", label="Binned Mean")
    if res is not None:
        axs[1].plot(res["x"], res["y_pred"], label="Loess Regression")
    axs[1].legend()

    # Show plot
    plt.show()

def cell_abundance_along_obs_val(
    adata: AnnData,
    obs_val: Union[str, Tuple[str, str]],
    groupby: Optional[str] = None,
    xlim: Tuple = (0, np.inf),
    savepath: Optional[os.PathLike] = None,
    figsize: Tuple = (8,6),
    save_only: bool = False,
    dpi_save: int = 300,
    multiple: Literal["layer", "dodge", "stack", "fill"] = "stack",
    histplot_element: Literal["bars", "step", "poly"] = "bars",
    kde: bool = False
    ):

    """
    Plot cell abundance along a specified observation value.

    Args:
        adata (AnnData): Annotated data matrix.
        obs_val (Union[str, Tuple[str, str]]): Observation category to be plotted on the x-axis.
            Can be a string representing a column in `adata.obs` or a tuple (obsm_key, obsm_col)
            where `obsm_key` is a key in `adata.obsm` and `obsm_col` is a column in the corresponding DataFrame.
        groupby (Optional[str], optional): Column in `adata.obs` to group by. Defaults to None.
        xmin (Number, optional): Minimum value of `obs_val` to include in the plot. Defaults to 0.
        savepath (Optional[os.PathLike], optional): Path to save the figure. Defaults to None.
        figsize (Tuple, optional): Size of the figure. Defaults to (8, 6).
        save_only (bool, optional): If True, only save the figure without displaying it. Defaults to False.
        dpi_save (int, optional): Dots per inch for saving the figure. Defaults to 300.
        histplot_multiple (str, optional): How to plot multiple histograms. Options are "layer", "dodge", "stack", "fill". Defaults to "stack".
        histplot_element (str, optional): Plotting element. Options are "bars", "step", "poly". Defaults to "bars".

    Returns:
        None
    """

    # check type of obs_val
    adata_obs = adata.obs.copy()
    if isinstance(obs_val, tuple):
        print("Retrieve `obs_val` from .obsm.")
        obsm_key = obs_val[0]
        obsm_col = obs_val[1]
        obs_val = f"distance_from_{obsm_col}"
        adata_obs[obs_val] = adata.obsm[obsm_key][obsm_col]

    # get data for plotting
    data = adata_obs[[obs_val, groupby]].dropna()

    # remove zeros
    xlim_mask = (data[obs_val] > xlim[0]) & (data[obs_val] <= xlim[1])
    data = data[xlim_mask].copy()

    # Create the histogram
    fig, ax = plt.subplots(1,1, figsize=(figsize[0], figsize[1]))

    if not kde:
        h = sns.histplot(data=data, x=obs_val,
                    hue=groupby, palette=DEFAULT_CATEGORICAL_CMAP.colors,
                    multiple=multiple, element=histplot_element,
                    alpha=1, ax=ax
                    )
    else:
        h = sns.kdeplot(data=data, x=obs_val,
                    hue=groupby, palette=DEFAULT_CATEGORICAL_CMAP.colors,
                    alpha=1, ax=ax, multiple=multiple
                    )
        plt.xlim(0, data[obs_val].max())

    # Move the legend outside of the plot
    sns.move_legend(h, "upper left", bbox_to_anchor=(1, 1))

    # save or show figure
    save_and_show_figure(savepath=savepath,
                         fig=h.get_figure(),
                         save_only=save_only,
                         dpi_save=dpi_save,
                         tight=True
                         )

# def cell_expression_along_axis(
#     adata,
#     obs_val,
#     genes,
#     cell_type_column,
#     cell_type,
#     xlim: Tuple[Number, Number] = (0, np.inf),
#     min_expression: Number = 0,
#     xlabel: Optional[str] = None,
#     fit_reg: bool = False,
#     lowess: bool = False,
#     robust: bool = False
#     ):
#     genes = convert_to_list(genes)

#     # if len(genes) == 1:
#     #     _single_cell_expression_along_axis(
#     #         adata=adata, obs_val=obs_val, gene=genes[0],
#     #         cell_type_column=cell_type_column, cell_type=cell_type,
#     #         xlim=xlim, min_expression=min_expression, xlabel=xlabel,
#     #         fit_reg=fit_reg, lowess=lowess, robust=robust
#     #     )
#     # elif len(genes) > 1:
#     _multi_cell_expression_along_axis(
#         adata=adata, obs_val=obs_val, genes=genes,
#         cell_type_column=cell_type_column, cell_type=cell_type,
#         xlim=xlim, min_expression=min_expression, xlabel=xlabel,
#         fit_reg=fit_reg, lowess=lowess, robust=robust
#     )
#     # else:
#     #     raise ValueError("`genes` must have length > 0.")

# def _single_cell_expression_along_axis(
#     adata,
#     obs_val,
#     gene,
#     cell_type_column,
#     cell_type,
#     xlim: Tuple[Number, Number] = (0, np.inf),
#     min_expression: Number = 0,
#     xlabel: Optional[str] = None,
#     fit_reg: bool = False,
#     lowess: bool = False,
#     robust: bool = False
#     ):

#     data_of_one_celltype = _select_data(
#         adata=adata,
#         obs_val=obs_val,
#         cell_type_column=cell_type_column,
#         cell_type=cell_type,
#         genes=gene,
#         xlim=xlim,
#     )

#     # Filter for minimum gene expression
#     data_of_one_celltype = data_of_one_celltype[data_of_one_celltype[gene] >= min_expression]

#     # Plot
#     g = sns.jointplot(data=data_of_one_celltype,
#                     x="axis", y=gene,
#                     height=4,
#                     color="firebrick", kind="kde", levels=8,
#                     marginal_kws={"fill": True},
#                     )
#     #g.plot_joint(sns.scatterplot, color="k", s=12)
#     g.plot_joint(sns.regplot, color="k",
#                  #lowess=True,
#                  fit_reg=fit_reg,
#                  lowess=lowess,
#                  robust=robust,
#                  scatter_kws={"s": 1},
#                  line_kws={"color": "orange"}
#                  )
#     g.ax_joint.set_ylabel(f"{gene} in '{cell_type}'")

#     # Set common x-label
#     if xlabel is None:
#         g.ax_joint.set_xlabel("_".join(convert_to_list(obs_val)))
#     else:
#         g.ax_joint.set_xlabel(xlabel)

#     # g = sns.jointplot(data=data_of_one_celltype,
#     #             x=axis_label, y=gene,
#     #             color="k", kind="reg", #levels=8,
#     #             marginal_kws={"fill": True},
#     #             )
#     # g.plot_marginals(sns.kdeplot, color="firebrick", #s=12
#     #                  )
#     plt.show()