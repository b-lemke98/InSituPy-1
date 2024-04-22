from typing import Literal, Optional

import numpy as np
import scanpy as sc
from parse import *
from scipy.sparse import csr_matrix

from insitupy import __version__
from insitupy.utils._scanorama import scanorama

from .._core._checks import check_raw


def normalize_anndata(adata,
              transformation_method: Literal["log1p", "sqrt"] = "log1p",
              verbose: bool = True
              ) -> None:
    # check if the matrix consists of raw integer counts
    check_raw(adata.X)

    # store raw counts in layer
    print("Store raw counts in anndata.layers['counts']...") if verbose else None
    adata.layers['counts'] = adata.X.copy()

    # preprocessing according to napari tutorial in squidpy
    print(f"Normalization, {transformation_method}-transformation...") if verbose else None
    sc.pp.normalize_total(adata)
    adata.layers['norm_counts'] = adata.X.copy()

    # transform either using log transformation or square root transformation
    if transformation_method == "log1p":
        sc.pp.log1p(adata)
    elif transformation_method == "sqrt":
        # Suggested in stlearn tutorial (https://stlearn.readthedocs.io/en/latest/tutorials/Xenium_PSTS.html)
        X = adata.X.toarray()
        adata.X = csr_matrix(np.sqrt(X) + np.sqrt(X + 1))
    else:
        raise ValueError(f'`transformation_method` is not one of ["log1p", "sqrt"]')

def reduce_dimensions_anndata(adata,
                              umap: bool = True,
                              tsne: bool = False,
                              batch_correction_key: Optional[str] = None,
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

    # clustering
    print("Leiden clustering...") if verbose else None
    sc.tl.leiden(adata)