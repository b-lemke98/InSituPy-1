from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from parse import *
from scipy.sparse import csr_matrix

from insitupy import __version__
from insitupy.utils._scanorama import scanorama

from .._core._checks import check_integer_counts


def normalize_and_transform_anndata(adata,
              transformation_method: Literal["log1p", "sqrt"] = "log1p",
              target_sum: int = None, # defaults to median of total counts of cells
              verbose: bool = True
              ) -> None:
    # check if the matrix consists of raw integer counts
    check_integer_counts(adata.X)

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


def test_transformation(adata, target_sum=1e4, layer=None):
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
    counts = _adata.X

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



def reduce_dimensions_anndata(adata,
                              umap: bool = True,
                              tsne: bool = False,
                              batch_correction_key: Optional[str] = None,
                              perform_clustering: bool = True,
                              verbose: bool = True,
                              tsne_lr: int = 1000,
                              tsne_jobs: int = 8,
                              **kwargs
                              ) -> None:
    """
    Reduce the dimensionality of the data using PCA, UMAP, and t-SNE techniques, optionally performing batch correction.

    Args:
        umap (bool, optional):
            If True, perform UMAP dimensionality reduction. Default is True.
        tsne (bool, optional):
            If True, perform t-SNE dimensionality reduction. Default is True.
        batch_correction_key (str, optional):
            Batch key for performing batch correction using scanorama. Default is None, indicating no batch correction.
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
    if batch_correction_key is None:
        # dimensionality reduction
        print("Dimensionality reduction...") if verbose else None
        sc.pp.pca(adata)
        if umap:
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
        if tsne:
            sc.tl.tsne(adata, n_jobs=tsne_jobs, learning_rate=tsne_lr)

    else:
        # PCA
        sc.pp.pca(adata)

        neigh_uncorr_key = 'neighbors_uncorrected'
        sc.pp.neighbors(adata, key_added=neigh_uncorr_key)

        if perform_clustering:
            # clustering
            sc.tl.leiden(adata, neighbors_key=neigh_uncorr_key, key_added='leiden_uncorrected')

        # batch correction
        print(f"Batch correction using scanorama for {batch_correction_key}...") if verbose else None
        hvgs = list(adata.var_names[adata.var['highly_variable']])
        adata = scanorama(adata, batch_key=batch_correction_key, hvg=hvgs, verbose=False, **kwargs)

        # find neighbors
        sc.pp.neighbors(adata, use_rep="X_scanorama")
        sc.tl.umap(adata)
        sc.tl.tsne(adata, use_rep="X_scanorama")

    if perform_clustering:
        # clustering
        print("Leiden clustering...") if verbose else None
        sc.tl.leiden(adata)