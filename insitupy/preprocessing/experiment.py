from __future__ import \
    annotations  # this prevents circular imports of type hints such as InSituExperiment in this case

from numbers import Number
from typing import Literal, Optional

import scanpy as sc
from parse import *
from tqdm import tqdm

from insitupy import __version__
from insitupy._core._checks import _is_experiment
from insitupy._core._utils import _get_cell_layer
from insitupy.preprocessing.anndata import (clustering_anndata,
                                            normalize_and_transform_anndata,
                                            reduce_dimensions_anndata)

from .._exceptions import ModalityNotFoundError


def calculate_qc_metrics(
    data: Optional[InSituExperiment, InSituData],
    cells_layer: Optional[str],
    percent_top: Number,
    log1p: bool,
    **kwargs
):
    is_experiment = _is_experiment(data)

    if is_experiment:
        iterator = tqdm(data.iterdata())
    else:
        iterator = zip([None], [data])

    for _, xd in iterator:
        cells = _get_cell_layer(cells=xd.cells, cells_layer=cells_layer)
        sc.pp.calculate_qc_metrics(
            cells.matrix, percent_top=percent_top, log1p=log1p, inplace=True, **kwargs
            )

def filter_cells(
    data: Optional[InSituExperiment, InSituData],
    cells_layer: Optional[str],
    min_counts: Optional[int] = None,
    min_genes: Optional[int] = None,
    max_counts: Optional[int] = None,
    max_genes: Optional[int] = None,
    **kwargs
):
    is_experiment = _is_experiment(data)

    if is_experiment:
        iterator = tqdm(data.iterdata())
    else:
        iterator = zip([None], [data])

    for _, xd in iterator:
        celldata = _get_cell_layer(cells=xd.cells, cells_layer=cells_layer)
        sc.pp.filter_cells(
            celldata.matrix,
            min_counts=min_counts,
            min_genes=min_genes,
            max_counts=max_counts,
            max_genes=max_genes,
            inplace=True,
            **kwargs
            )

        # sync cell names between boundaries and matrix
        celldata.sync()

def normalize_and_transform(
    data: Optional[InSituExperiment, InSituData],
    cells_layer: Optional[str],
    adata_layer: Optional[str] = None,
    transformation_method: Literal["log1p", "sqrt"] = "log1p",
    target_sum: int = 250,
    scale: bool = False,
    assert_integer_counts: bool = True,
    verbose: bool = False
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
    is_experiment = _is_experiment(data)

    if is_experiment:
        iterator = tqdm(data.iterdata())
    else:
        iterator = zip([None], [data])

    for _, xd in iterator:
        if xd.cells is not None:
            celldata = _get_cell_layer(cells=xd.cells, cells_layer=cells_layer)
            normalize_and_transform_anndata(
                adata=celldata.matrix,
                layer=adata_layer,
                transformation_method=transformation_method,
                target_sum=target_sum,
                scale=scale,
                verbose=verbose,
                assert_integer_counts=assert_integer_counts
                )
        else:
            raise ModalityNotFoundError(modality="cells")

def reduce_dimensions(
    data: Optional[InSituExperiment, InSituData],
    cells_layer: Optional[str],
    method: Literal["umap", "tsne"] = "umap",
    n_neighbors: int = 16,
    n_pcs: int = 0,
    ):
    """
    Performs dimensionality reduction of the data using either UMAP or TSNE.

    Args:
        data (Optional[InSituExperiment, InSituData]): The experiment or sample-level data object containing cell information.
        method (Literal["umap", "tsne"], optional): The dimensionality reduction method to use. Defaults to "umap".
        cells_layer (Optional[str]): The specific layer of cells to use for reduction.
        n_neighbors (int, optional): The number of neighbors to use in the reduction method. Defaults to 16.
        n_pcs (int, optional): The number of principal components to use. Defaults to 0.

    Raises:
        ModalityNotFoundError: If the 'cells' modality is not found in the individual samples.

    """

    is_experiment = _is_experiment(data)

    if is_experiment:
        iterator = tqdm(data.iterdata())
    else:
        iterator = zip([None], [data])

    for _, xd in iterator:
        if xd.cells is not None:
            celldata = _get_cell_layer(cells=xd.cells, cells_layer=cells_layer)

            reduce_dimensions_anndata(
                adata=celldata.matrix,
                method=method,
                n_neighbors=n_neighbors,
                n_pcs=n_pcs
                )
        else:
            raise ModalityNotFoundError(modality="cells")

def clustering(
    data: Optional[InSituExperiment, InSituData],
    cells_layer: Optional[str],
    method: Literal["leiden", "louvain"] = "leiden",
    verbose: bool = True
    ):
    """
    Performs clustering on the data using the specified method.

    Args:
        data (Optional[InSituExperiment, InSituData]): The experiment or sample-level data object containing cell information.
        cells_layer (Optional[str]): The specific layer of cells to use for clustering.
        method (Literal["leiden", "louvain"], optional): The clustering method to use. Defaults to "leiden".
        verbose (bool, optional): If True, enables verbose output. Defaults to True.

    Raises:
        ModalityNotFoundError: If the 'cells' modality is not found in the individual samples.

    """
    is_experiment = _is_experiment(data)

    if is_experiment:
        iterator = tqdm(data.iterdata())
    else:
        iterator = zip([None], [data])

    for _, xd in iterator:
        if xd.cells is not None:
            celldata = _get_cell_layer(cells=xd.cells, cells_layer=cells_layer)

            clustering_anndata(
                adata=celldata.matrix,
                method=method,
                verbose=False
                )
        else:
            raise ModalityNotFoundError(modality="cells")