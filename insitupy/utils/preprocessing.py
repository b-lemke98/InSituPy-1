from numbers import Number
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from parse import *
from scipy.sparse import csr_matrix

from insitupy import __version__
from insitupy._core._checks import check_integer_counts
from insitupy.utils.utils import textformat as tf


def normalize_and_transform_anndata(adata,
              transformation_method: Literal["log1p", "sqrt"] = "log1p",
              target_sum: int = None, # defaults to median of total counts of cells
              scale: bool = True,
              verbose: bool = True
              ) -> None:
    # check if the matrix consists of raw integer counts
    #check_integer_counts(adata.X)

    # store raw counts in layer
    print("Store raw counts in anndata.layers['counts']...") if verbose else None
    adata.layers['counts'] = adata.X.copy()

    # preprocessing according to napari tutorial in squidpy
    print(f"Normalization, {transformation_method}-transformation...") if verbose else None
    sc.pp.normalize_total(adata, target_sum=target_sum)
    adata.layers['norm_counts'] = adata.X.copy()

    # transform either using log transformation or square root transformation
    if transformation_method == "log1p":
        sc.pp.log1p(adata)
    elif transformation_method == "sqrt":
        # Suggested in stlearn tutorial (https://stlearn.readthedocs.io/en/latest/tutorials/Xenium_PSTS.html)
        try:
            X = adata.X.toarray()
        except AttributeError:
            X = adata.X
        adata.X = csr_matrix(np.sqrt(X) + np.sqrt(X + 1))
    else:
        raise ValueError(f'`transformation_method` is not one of ["log1p", "sqrt"]')

    if scale:
        sc.pp.scale(adata)


def test_transformation(
    adata: AnnData,
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

    # create a copy of the anndata
    _adata = adata.copy()

    # Check if the matrix consists of raw integer counts
    if layer is None:
        check_integer_counts(_adata.X)
    else:
        _adata.X = _adata.layers[layer].copy()
        check_integer_counts(_adata.X)

    # get raw counts
    raw_counts = _adata.X.copy()

    # Preprocessing according to napari tutorial in squidpy
    sc.pp.normalize_total(_adata, target_sum=target_sum)

    # Create a copy of the anndata object for log1p transformation
    adata_log1p = _adata.copy()
    sc.pp.log1p(adata_log1p)

    # Create a copy of the anndata object for sqrt transformation
    adata_sqrt = _adata.copy()
    try:
        X = adata_sqrt.X.toarray()
    except AttributeError:
        X = adata_sqrt.X
    adata_sqrt.X = np.sqrt(X) + np.sqrt(X + 1)

    if scale:
        sc.pp.scale(adata_log1p)
        sc.pp.scale(adata_sqrt)

    # Plot histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].hist(raw_counts.sum(axis=1), bins=50, color='skyblue', edgecolor='black')
    axes[0].set_title('Raw Counts', fontsize=14)
    axes[0].set_xlabel('Counts', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)

    axes[1].hist(adata_log1p.X.sum(axis=1), bins=50, color='skyblue', edgecolor='black')
    axes[1].set_title('Log1p Transformed Counts', fontsize=14)
    axes[1].set_xlabel('Counts', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)

    axes[2].hist(adata_sqrt.X.sum(axis=1), bins=50, color='skyblue', edgecolor='black')
    axes[2].set_title('Sqrt Transformed Counts', fontsize=14)
    axes[2].set_xlabel('Counts', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)


    plt.tight_layout()
    plt.show()


def plot_qc_metrics(adata):
    """
    Plots the QC metrics calculated by sc.pp.calculate_qc_metrics.

    Parameters:
    adata : AnnData
        Annotated data matrix with QC metrics calculated.
    """
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

    fig, axes = plt.subplots(2, max(len(obs_metrics), len(var_metrics)), figsize=(20, 10))

    # Add titles to each row
    if len(obs_metrics) > 0:
        axes[0, 0].annotate('.obs Metrics', xy=(0, 0.5), xytext=(-axes[0, 0].yaxis.labelpad - 5, 0),
                            xycoords=axes[0, 0].yaxis.label, textcoords='offset points',
                            size='large', ha='right', va='center', rotation=90, weight='bold')

    if len(var_metrics) > 0:
        axes[1, 0].annotate('.var Metrics', xy=(0, 0.5), xytext=(-axes[1, 0].yaxis.labelpad - 5, 0),
                            xycoords=axes[1, 0].yaxis.label, textcoords='offset points',
                            size='large', ha='right', va='center', rotation=90, weight='bold')

    for i, metric in enumerate(obs_metrics):
        sns.histplot(adata.obs[metric], bins=50, color='skyblue', edgecolor='black', kde=False, ax=axes[0, i])
        axes[0, i].set_title(metric)
        axes[0, i].set_xlabel('Value')
        axes[0, i].set_ylabel('Frequency')

    for i, metric in enumerate(var_metrics):
        sns.histplot(adata.var[metric], bins=50, color='skyblue', edgecolor='black', kde=False, ax=axes[1, i])
        axes[1, i].set_title(metric)
        axes[1, i].set_xlabel('Value')
        axes[1, i].set_ylabel('Frequency')

    # Remove empty subplots
    if len(obs_metrics) < len(var_metrics):
        for j in range(len(obs_metrics), len(var_metrics)):
            fig.delaxes(axes[0, j])
    elif len(var_metrics) < len(obs_metrics):
        for j in range(len(var_metrics), len(obs_metrics)):
            fig.delaxes(axes[1, j])

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()



def reduce_dimensions_anndata(adata,
                              umap: bool = True,
                              tsne: bool = False,
                              perform_clustering: bool = True,
                              tsne_lr: int = 1000,
                              tsne_jobs: int = 8,
                              n_neighbors: int = 16,
                              n_pcs: int = 0,
                              leiden: bool = True,
                              louvain: bool = True,
                              verbose: bool = True,
                              **kwargs
                              ) -> None:
    """
    Reduce the dimensionality of the data using PCA, UMAP, and t-SNE techniques, optionally performing batch correction.

    Args:
        umap (bool, optional):
            If True, perform UMAP dimensionality reduction. Default is True.
        tsne (bool, optional):
            If True, perform t-SNE dimensionality reduction. Default is True.
        verbose (bool, optional):
            If True, print progress messages during dimensionality reduction. Default is True.
        tsne_lr (int, optional):
            Learning rate for t-SNE. Default is 1000.
        tsne_jobs (int, optional):
            Number of CPU cores to use for t-SNE computation. Default is 8.
        **kwargs:
            Additional keyword arguments to be passed to scanorama function if batch correction is performed.

    Raises:
        ValueError: If an invalid `batch_correction_key` is provided.

    Returns:
        None: This method modifies the input matrix in place, reducing its dimensionality using specified techniques and
            batch correction if applicable. It does not return any value.
    """
    # dimensionality reduction
    print("Dimensionality reduction...") if verbose else None
    sc.pp.pca(adata)
    if umap:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        sc.tl.umap(adata)
    if tsne:
        sc.tl.tsne(adata, n_jobs=tsne_jobs, learning_rate=tsne_lr)

    if perform_clustering:
        # clustering
        if leiden:
            print("Leiden clustering...") if verbose else None
            sc.tl.leiden(adata)
        if louvain:
            print("Louvain clustering...") if verbose else None
            sc.tl.louvain(adata)