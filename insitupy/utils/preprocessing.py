import numpy as np
import anndata
import scanpy as sc
from .utils import split_batches, check_sanity
from typing import Optional, Tuple, Union, List, Dict, Any, Literal
from scipy.sparse import issparse, csr_matrix
from .utils import check_raw


def normalize(self,
              transformation_method: Literal["log1p", "sqrt"] = "log1p",
              verbose: bool = True
              ) -> None:
    """
    Normalize the data using either log1p or square root transformation.

    Args:
        transformation_method (Literal["log1p", "sqrt"], optional):
            The method used for data transformation. Choose between "log1p" for logarithmic transformation
            and "sqrt" for square root transformation. Default is "log1p".
        verbose (bool, optional):
            If True, print progress messages during normalization. Default is True.

    Raises:
        ValueError: If `transformation_method` is not one of ["log1p", "sqrt"].

    Returns:
        None: This method modifies the input matrix in place, normalizing the data based on the specified method.
              It does not return any value.
    """
    # check if the matrix consists of raw integer counts
    check_raw(self.matrix.X)

    # store raw counts in layer
    print("Store raw counts in anndata.layers['counts']...") if verbose else None
    self.matrix.layers['counts'] = self.matrix.X.copy()

    # preprocessing according to napari tutorial in squidpy
    print(f"Normalization, {transformation_method}-transformation...") if verbose else None
    sc.pp.normalize_total(self.matrix)
    self.matrix.layers['norm_counts'] = self.matrix.X.copy()
    
    # transform either using log transformation or square root transformation
    if transformation_method == "log1p":
        sc.pp.log1p(self.matrix)
    elif transformation_method == "sqrt":
        # Suggested in stlearn tutorial (https://stlearn.readthedocs.io/en/latest/tutorials/Xenium_PSTS.html)
        X = self.matrix.X.toarray()   
        self.matrix.X = csr_matrix(np.sqrt(X) + np.sqrt(X + 1))
    else:
        raise ValueError(f'`transformation_method` is not one of ["log1p", "sqrt"]')
    
    
def hvg(self,
        hvg_batch_key: Optional[str] = None, 
        hvg_flavor: Literal["seurat", "cell_ranger", "seurat_v3"] = 'seurat', 
        hvg_n_top_genes: Optional[int] = None,
        verbose: bool = True
        ) -> None:
    """
    Calculate highly variable genes (HVGs) using specified flavor and parameters.

    Args:
        hvg_batch_key (str, optional):
            Batch key for computing HVGs separately for each batch. Default is None, indicating all samples are considered.
        hvg_flavor (Literal["seurat", "cell_ranger", "seurat_v3"], optional):
            Flavor of the HVG computation method. Choose between "seurat", "cell_ranger", or "seurat_v3".
            Default is 'seurat'.
        hvg_n_top_genes (int, optional):
            Number of top highly variable genes to identify. Mandatory if `hvg_flavor` is set to "seurat_v3".
            Default is None.
        verbose (bool, optional):
            If True, print progress messages during HVG computation. Default is True.

    Raises:
        ValueError: If `hvg_n_top_genes` is not specified for "seurat_v3" flavor or if an invalid `hvg_flavor` is provided.

    Returns:
        None: This method modifies the input matrix in place, identifying highly variable genes based on the specified
              flavor and parameters. It does not return any value.
    """
    
    if hvg_flavor in ["seurat", "cell_ranger"]:
        hvg_layer = None
    elif hvg_flavor == "seurat_v3":
        hvg_layer = "counts" # seurat v3 method expects counts data

        # n top genes must be specified for this method
        if hvg_n_top_genes is None:
            raise ValueError(f"HVG computation: For flavor {hvg_flavor} `hvg_n_top_genes` is mandatory")
    else:
        raise ValueError(f'Unknown value for `hvg_flavor`: {hvg_flavor}. Possible values: {["seurat", "cell_ranger", "seurat_v3"]}')

    if hvg_batch_key is None:
        print("Calculate highly-variable genes across all samples using {} flavor...".format(hvg_flavor)) if verbose else None
    else:
        print("Calculate highly-variable genes per batch key {} using {} flavor...".format(hvg_batch_key, hvg_flavor)) if verbose else None

    sc.pp.highly_variable_genes(self.matrix, batch_key=hvg_batch_key, flavor=hvg_flavor, layer=hvg_layer, n_top_genes=hvg_n_top_genes)

        
def reduce_dimensions(self,
                      umap: bool = True, 
                      tsne: bool = True,
                      batch_correction_key: Optional[str] = None,
                      verbose: bool = True,
                      tsne_lr: int = 1000, 
                      tsne_jobs: int = 8,
                      **kwargs
                      ):
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
        sc.pp.pca(self.matrix)
        if umap:
            sc.pp.neighbors(self.matrix)
            sc.tl.umap(self.matrix)
        if tsne:
            sc.tl.tsne(self.matrix, n_jobs=tsne_jobs, learning_rate=tsne_lr)

    else:
        # PCA
        sc.pp.pca(self.matrix)

        neigh_uncorr_key = 'neighbors_uncorrected'
        sc.pp.neighbors(self.matrix, key_added=neigh_uncorr_key)

        # clustering
        sc.tl.leiden(self.matrix, neighbors_key=neigh_uncorr_key, key_added='leiden_uncorrected')  

        # batch correction
        print(f"Batch correction using scanorama for {batch_correction_key}...") if verbose else None
        hvgs = list(self.matrix.var_names[self.matrix.var['highly_variable']])
        self.matrix = scanorama(self.matrix, batch_key=batch_correction_key, hvg=hvgs, verbose=False, **kwargs)

        # find neighbors
        sc.pp.neighbors(self.matrix, use_rep="X_scanorama")
        sc.tl.umap(self.matrix)
        sc.tl.tsne(self.matrix, use_rep="X_scanorama")

    # clustering
    print("Leiden clustering...") if verbose else None
    sc.tl.leiden(self.matrix)


def scanorama(adata, 
              batch_key: str, 
              hvg: bool = False, 
              hvg_key: str = 'highly_variable', 
              **kwargs
              ):
    """
    Perform Scanorama batch correction on the input AnnData object.

    Scanorama is a batch correction method for single-cell RNA-seq data.
    For more details, see: https://github.com/brianhie/scanorama/

    Args:
        adata (anndata.AnnData):
            Annotated data matrix, where rows correspond to cells and columns correspond to features.
        batch_key (str):
            Batch key specifying the batch labels for each cell in `adata.obs`.
        hvg (bool, optional):
            If True, use highly variable genes for batch correction. Default is False.
        hvg_key (str, optional):
            Key in `adata.var` indicating highly variable genes if `hvg` is True. Default is 'highly_variable'.
        **kwargs:
            Additional keyword arguments to be passed to scanorama.correct_scanpy function.

    Returns:
        anndata.AnnData: 
            Annotated data matrix with batch-corrected values.
            The original data remains unchanged, and the batch-corrected values can be accessed using
            `adata.obsm['X_scanorama']`.
    """
    import scanorama

    check_sanity(adata, batch_key, hvg, hvg_key)

    hvg_genes = list(adata.var.index[adata.var[hvg_key]])

    split, categories = split_batches(adata.copy(), batch_key, hvg=hvg_genes, return_categories=True)
    corrected = scanorama.correct_scanpy(split, return_dimred=True, **kwargs)
    corrected = anndata.AnnData.concatenate(
        *corrected, batch_key=batch_key, batch_categories=categories, index_unique=None
    )
    corrected.obsm['X_emb'] = corrected.obsm['X_scanorama']
    # corrected.uns['emb']=True

    # add scanorama results to original adata - make sure to have correct order of obs
    X_scan = corrected.obsm['X_scanorama']
    orig_obs_names = list(adata.obs_names)
    cor_obs_names = list(corrected.obs_names)
    adata.obsm['X_scanorama'] = np.array([X_scan[orig_obs_names.index(o)] for o in cor_obs_names])
    adata.obsm['X_emb'] = adata.obsm['X_scanorama']

    return adata