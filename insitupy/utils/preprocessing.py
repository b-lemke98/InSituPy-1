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
              ):
    '''
    Function to normalize data.
    '''
    # check if the matrix consists of raw integer counts
    check_raw(self.matrix.X)

    # store raw counts in layer
    print("Store raw counts in anndata.layers['counts']...") if verbose else None
    self.matrix.layers['counts'] = self.matrix.X.copy()

    # preprocessing according to napari tutorial in squidpy
    print("Normalization, log-transformation...") if verbose else None
    sc.pp.normalize_total(self.matrix)
    self.matrix.layers['norm_counts'] = self.matrix.X.copy()
    
    # normalize either 
    if transformation_method == "log1p":
        sc.pp.log1p(self.matrix)
    elif transformation_method == "sqrt":
        X = self.matrix.X.toarray()   
        self.matrix.X = csr_matrix(np.sqrt(X) + np.sqrt(X + 1))
    
    
def hvg(self,
        hvg_batch_key: Optional[str] = None, 
        hvg_flavor: str = 'seurat', 
        hvg_n_top_genes: Optional[int] = None,
        verbose: bool = True
        ):
    
    '''
    Function to calculate highly variable genes.
    '''
    
    if hvg_flavor in ["seurat", "cell_ranger"]:
        hvg_layer = None
    elif hvg_flavor == "seurat_v3":
        hvg_layer = "counts" # seurat v3 method expects counts data

        # n top genes must be specified for this method
        if hvg_n_top_genes is None:
            print("HVG computation: For flavor {} `hvg_n_top_genes` is mandatory".format(hvg_flavor)) if verbose else None
            return
    else:
        print("Unknown value for `hvg_flavor`: {}. Possible values: {}".format(hvg_flavor, ["seurat", "cell_ranger", "seurat_v3"])) if verbose else None

    if hvg_batch_key is None:
        print("Calculate highly-variable genes across all samples using {} flavor...".format(hvg_flavor)) if verbose else None
    else:
        print("Calculate highly-variable genes per batch key {} using {} flavor...".format(hvg_batch_key, hvg_flavor)) if verbose else None

    sc.pp.highly_variable_genes(self.matrix, batch_key=hvg_batch_key, flavor=hvg_flavor, layer=hvg_layer, n_top_genes=hvg_n_top_genes)

        
def reduce_dimensions(self,
                      umap: bool = True, 
                      tsne: bool = True,
                      batch_correction_key: Optional[str] = None,
                      batch_correction_method: str = "scanorama", 
                      verbose: bool = True,
                      tsne_lr: int = 1000, 
                      tsne_jobs: int = 8,
                      **kwargs
                      ):
    
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
        print("Batch correction using {} for {}...".format(batch_correction_method, batch_correction_key)) if verbose else None
        hvgs = list(self.matrix.var_names[self.matrix.var['highly_variable']])
        self.matrix = scanorama(self.matrix, batch=batch_correction_key, hvg=hvgs, verbose=False, **kwargs)

        # find neighbors
        sc.pp.neighbors(self.matrix, use_rep="X_scanorama")
        sc.tl.umap(self.matrix)
        sc.tl.tsne(self.matrix, use_rep="X_scanorama")

    # clustering
    print("Leiden clustering...") if verbose else None
    sc.tl.leiden(self.matrix)


def scanorama(adata, batch, hvg=False, hvg_key='highly_variable', **kwargs):

    '''
    Function to perform Scanorama batch correction (https://github.com/brianhie/scanorama/).
    Code partially from: https://github.com/theislab/scib.
    '''

    import scanorama

    check_sanity(adata, batch, hvg, hvg_key)

    hvg_genes = list(adata.var.index[adata.var[hvg_key]])

    split, categories = split_batches(adata.copy(), batch, hvg=hvg_genes, return_categories=True)
    corrected = scanorama.correct_scanpy(split, return_dimred=True, **kwargs)
    corrected = anndata.AnnData.concatenate(
        *corrected, batch_key=batch, batch_categories=categories, index_unique=None
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