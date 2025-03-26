from __future__ import \
    annotations  # this prevents circular imports of type hints such as InSituExperiment in this case

from numbers import Number
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData

from insitupy._core._checks import check_integer_counts
from insitupy._core._utils import _get_cell_layer


def plot_qc_metrics(
    data: InSituData, # type: ignore #TODO: Expand this to InSituExperiment
    cells_layer: Optional[str] = None,
    ):
    """
    Plots the QC metrics calculated by sc.pp.calculate_qc_metrics.

    Parameters:
    adata : AnnData
        Annotated data matrix with QC metrics calculated.
    """
    # retrieve AnnData from cell layer
    celldata = _get_cell_layer(cells=data.cells, cells_layer=cells_layer)
    adata = celldata.matrix

    # QC metrics in .obs
    obs_metrics = ['total_counts', 'n_genes_by_counts', 'pct_counts_mt']
    # QC metrics in .var
    var_metrics = ['n_cells_by_counts', 'mean_counts', 'pct_dropout_by_counts', 'total_counts']

    # Check if all metrics exist in .obs
    obs_metrics = [metric for metric in obs_metrics if metric in adata.obs]
    if len(obs_metrics) == 0:
        print("Warning: No .obs metrics found in adata.obs")

    # Check if all metrics exist in .var
    var_metrics = [metric for metric in var_metrics if metric in adata.var]
    if len(var_metrics) == 0:
        print("Warning: No .var metrics found in adata.var")

    ncols = len(obs_metrics+var_metrics)
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))

    # Add titles to each row
    if len(obs_metrics) > 0:
        axes[0].annotate('.obs Metrics', xy=(0, 0.5), xytext=(-axes[0].yaxis.labelpad - 5, 0),
                            xycoords=axes[0].yaxis.label, textcoords='offset points',
                            size='large', ha='right', va='center', rotation=90, weight='bold')

    if len(var_metrics) > 0:
        axes[2].annotate('.var Metrics', xy=(0, 0.5), xytext=(-axes[2].yaxis.labelpad - 5, 0),
                            xycoords=axes[2].yaxis.label, textcoords='offset points',
                            size='large', ha='right', va='center', rotation=90, weight='bold')

    for i, metric in enumerate(obs_metrics):
        sns.histplot(adata.obs[metric], bins=50, color='skyblue', edgecolor='black', kde=False, ax=axes[i])
        axes[i].set_title(metric)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')

    for i, metric in enumerate(var_metrics):
        sns.histplot(adata.var[metric], bins=50, color='coral', edgecolor='black', kde=False, ax=axes[2+i])
        axes[2+i].set_title(metric)
        axes[2+i].set_xlabel('Value')
        axes[2+i].set_ylabel('Frequency')

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()


def test_transformations(
    data: InSituData, # type: ignore #TODO: Expand this to InSituExperiment
    cells_layer: Optional[str] = None,
    target_sum: Number = 250,
    layer: Optional[str] = None,
    scale: bool = False
        ):
    """
    Test normalization and transformation methods by plotting histograms of raw,
    log1p-transformed, and sqrt-transformed counts.

    Args:
        adata (AnnData): Annotated data matrix.
        target_sum (int, optional): Target sum for normalization. Defaults to 1e4.
        layer (str, optional): Layer to use for transformation. Defaults to None.
    """
    # retrieve AnnData from cell layer
    celldata = _get_cell_layer(cells=data.cells, cells_layer=cells_layer)
    adata = celldata.matrix.copy() # copy it to not affect it during the plotting

    # Check if the matrix consists of raw integer counts
    if layer is None:
        check_integer_counts(adata.X)
    else:
        adata.X = adata.layers[layer].copy()
        check_integer_counts(adata.X)

    # get raw counts
    raw_counts = adata.X.copy()

    # Preprocessing according to napari tutorial in squidpy
    sc.pp.normalize_total(adata, target_sum=target_sum)

    # Create a copy of the anndata object for log1p transformation
    adata_log1p = adata.copy()
    sc.pp.log1p(adata_log1p)

    # Create a copy of the anndata object for sqrt transformation
    adata_sqrt = adata.copy()
    try:
        X = adata_sqrt.X.toarray()
    except AttributeError:
        X = adata_sqrt.X
    adata_sqrt.X = np.sqrt(X) + np.sqrt(X + 1)

    if scale:
        sc.pp.scale(adata_log1p)
        sc.pp.scale(adata_sqrt)
        titles = ['Log1p-transformed and scaled counts','Sqrt-transformed and scaled counts']
    else:
        titles = ['Log1p-transformed counts','Sqrt-transformed ounts']

    # Plot histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].hist(raw_counts.sum(axis=1), bins=50, color='skyblue', edgecolor='black')
    axes[0].set_title('Raw Counts', fontsize=14)
    axes[0].set_xlabel('Counts', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)

    axes[1].hist(adata_log1p.X.sum(axis=1), bins=50, color='skyblue', edgecolor='black')
    axes[1].set_title(titles[0], fontsize=14)
    axes[1].set_xlabel('Counts', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)

    axes[2].hist(adata_sqrt.X.sum(axis=1), bins=50, color='skyblue', edgecolor='black')
    axes[2].set_title(titles[1], fontsize=14)
    axes[2].set_xlabel('Counts', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)


    plt.tight_layout()
    plt.show()