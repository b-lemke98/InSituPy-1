import os
import warnings
from numbers import Number
from pathlib import Path
from typing import List, Literal, Optional, Union

import dask.array as da
import pandas as pd
import scanpy as sc
import zarr
from anndata import AnnData
from pandas.api.types import is_numeric_dtype
from parse import *
from scipy.sparse import csr_matrix
from zarr.errors import ArrayNotFoundError

from insitupy import __version__
from insitupy._core.dataclasses import (AnnotationsData, BoundariesData,
                                        CellData, RegionsData)
from insitupy._exceptions import InvalidFileTypeError
from insitupy.utils.io import read_json
from insitupy.utils.utils import decode_robust_series

from ..utils.io import read_json
from ..utils.utils import decode_robust_series
from .dataclasses import AnnotationsData, BoundariesData, CellData, RegionsData


def read_celldata(
    path: Union[str, os.PathLike, Path],
    pixel_size: Number
    ) -> CellData:
    # read metadata
    path = Path(path)
    celldata_metadata = read_json(path / ".celldata")

    # read matrix data
    matrix = sc.read_h5ad(path / celldata_metadata["matrix"])
    
    # get path of boundaries data
    bound_path = path / celldata_metadata["boundaries"]
    
    # read cell ids and seg_mask_values
    cell_ids = da.from_zarr(bound_path, component="cell_id")
    
    try:
        # in older datasets sometimes seg_mask_value is missing
        seg_mask_value = da.from_zarr(bound_path, component="seg_mask_value")
    except ArrayNotFoundError:
        seg_mask_value = None
    
    # create boundaries data object
    boundaries = BoundariesData(cell_ids=cell_ids, seg_mask_value=seg_mask_value)

    # retrieve the boundaries data
    bound_data = {}
    meta = {}
    with zarr.ZipStore(bound_path, mode="r") as zipstore:
        for k in zipstore.listdir("masks"):
            if not k.startswith("."):
                # iterate through subresolutions
                subresolutions = zipstore.listdir(f"masks/{k}")
                
                if ".zarray" in subresolutions:
                    bound_data[k] = da.from_zarr(zipstore).persist()
                else:
                    # it is stored as pyramid -> initialize a list for the pyramid
                    bound_data[k] = []
                    for subres in subresolutions:
                        if not subres.startswith("."):
                            # append the pyramid to the list
                            bound_data[k].append(da.from_zarr(zipstore, component=f"masks/{k}/{subres}").persist())
                
                # retrieve boundaries metadata
                store = zarr.open(zipstore)
                meta[k] = store[f"masks/{k}"].attrs.asdict()
    
    # add boundaries
    boundaries.add_boundaries(data=bound_data,
                              pixel_size=meta[k]["pixel_size"]
                              )

    # create CellData object
    celldata = CellData(matrix=matrix, boundaries=boundaries)

    return celldata


def read_regionsdata(
    path: Union[str, os.PathLike, Path],
):
    metadata = read_json(path / "metadata.json")
    keys = metadata.keys()
    files = [path / f"{k}.geojson" for k in keys]
    data = RegionsData(files, keys)

    # overwrite metadata
    data.metadata = metadata
    return data


def read_annotationsdata(
    path: Union[str, os.PathLike, Path],
):
    path = Path(path)
    metadata = read_json(path / "metadata.json")
    keys = metadata.keys()
    files = [path / f"{k}.geojson" for k in keys]
    data = AnnotationsData(files, keys)

    # overwrite metadata
    data.metadata = metadata
    return data


def _read_matrix_from_xenium(path) -> AnnData:
    # extract parameters from metadata
    cf_h5_path = path / "cell_feature_matrix.h5"

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        # read matrix data
        adata = sc.read_10x_h5(cf_h5_path)

    # read cell information
    cells_parquet_path = path / "cells.parquet"
    cells = pd.read_parquet(cells_parquet_path)

    # transform cell ids from bytes to str
    cells = cells.set_index("cell_id")

    # make sure that the indices are decoded strings
    if is_numeric_dtype(cells.index):
        cells.index = cells.index.astype(str)
    else:
        cells.index = decode_robust_series(cells.index)

    # add information to anndata observations
    adata.obs = pd.merge(left=adata.obs, right=cells, left_index=True, right_index=True)

    # transfer coordinates to .obsm
    coord_cols = ["x_centroid", "y_centroid"]
    adata.obsm["spatial"] = adata.obs[coord_cols].values
    adata.obsm["spatial"]
    adata.obs.drop(coord_cols, axis=1, inplace=True)

    return adata


def _read_boundaries_from_xenium(
    path: Union[str, os.PathLike, Path],
    pixel_size: Number = 1,
    mode: Literal["dataframe", "mask"] = "mask"
    ) -> BoundariesData:
    # # read boundaries data
    path = Path(path)

    if mode == "dataframe":
        files=["cell_boundaries.parquet", "nucleus_boundaries.parquet"]
        labels=["cellular", "nuclear"]

        # generate path for files
        files = [path / f for f in files]

        # generate dataframes
        data_dict = {}
        for n, f in zip(labels, files):
            # check the file suffix
            if not f.suffix == ".parquet":
                InvalidFileTypeError(allowed_types=[".parquet"], received_type=f.suffix)

            # load dataframe
            df = pd.read_parquet(f)

            # decode columns
            df = df.apply(lambda x: decode_robust_series(x), axis=0)

            # collect dataframe
            data_dict[n] = df
            
        # create boundariesdata object
        boundaries = BoundariesData()

    else:
        cells_zarr_file = path / "cells.zarr.zip"

        # open zarr directory using dask
        data_dict = {
            "nuclear": da.from_zarr(cells_zarr_file, component="masks/0"),
            "cellular": da.from_zarr(cells_zarr_file, component="masks/1")
        }
        
        # read cell ids and seg mask value
        # for info see: https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/analysis/xoa-output-zarr#cells
        cell_ids = da.from_zarr(cells_zarr_file, component="cell_id")
        
        try:
            seg_mask_value = da.from_zarr(cells_zarr_file, component="seg_mask_value")
        except ArrayNotFoundError:
            seg_mask_value = None
            
        # create boundariesdata object
        boundaries = BoundariesData(cell_ids=cell_ids, seg_mask_value=seg_mask_value)

    boundaries.add_boundaries(data=data_dict, 
                              pixel_size=pixel_size)

    return boundaries


def _read_binned_expression(
    path: Union[str, os.PathLike, Path],
    gene_names_to_select = List
):
    # add binned expression data to .varm of self.cells.matrix
    trans_file = path / "transcripts.zarr.zip"

    # read zarr store
    t = zarr.open(trans_file, mode="r")

    # extract sparse array
    data_gene = t["density/gene"]
    data = data_gene["data"][:]
    indices = data_gene["indices"][:]
    indptr = data_gene["indptr"][:]

    # get dimensions of the array
    cols = data_gene.attrs["cols"]
    rows = data_gene.attrs["rows"]

    # get info on gene names
    gene_names = data_gene.attrs["gene_names"]
    n_genes = len(gene_names)

    sarr = csr_matrix((data, indices, indptr))

    # reshape to get binned data
    arr = sarr.toarray()
    arr = arr.reshape((n_genes, rows, cols))

    # select only genes that are available in the adata object
    gene_mask = [elem in gene_names_to_select for elem in gene_names]
    arr = arr[gene_mask]
    return arr


