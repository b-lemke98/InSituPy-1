import functools as ft
import gc
import json
import os
import shutil
import warnings
from datetime import datetime
from math import ceil
from numbers import Number
from os.path import abspath
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union
from uuid import uuid4

import dask
import dask.array as da
import dask_image
import matplotlib
import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import zarr
from anndata import AnnData
from geopandas import GeoDataFrame
from napari.layers import Layer, Shapes
from napari.layers.shapes.shapes import Shapes
from pandas.api.types import is_numeric_dtype
from parse import *
from rasterio.features import rasterize
from scipy.sparse import csr_matrix, issparse
from shapely import Point, Polygon, affinity
from shapely.geometry.polygon import Polygon

from insitupy import __version__

from .._constants import CACHE
from .._exceptions import (InvalidFileTypeError, ModalityNotFoundError,
                           NotOneElementError, UnknownOptionError,
                           WrongNapariLayerTypeError, XeniumDataMissingObject,
                           XeniumDataRepeatedCropError)
from ..image import ImageRegistration, deconvolve_he, resize_image
from ..image.utils import create_img_pyramid
from ..utils.io import (check_overwrite_and_remove_if_true,
                        read_baysor_polygons, read_json, write_dict_to_json)
from ..utils.utils import convert_to_list, decode_robust_series
from ..utils.utils import textformat as tf
from ._checks import check_raw, check_zip
from ._layers import _add_annotations_as_layer
from ._save import (_save_alt, _save_annotations, _save_cells, _save_images,
                    _save_regions, _save_transcripts)
from ._scanorama import scanorama
from ._widgets import (_annotation_widget, _create_points_layer,
                       _initialize_widgets)
from .dataclasses import (AnnotationsData, BoundariesData, CellData, ImageData,
                          RegionsData)


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

def _read_boundaries_from_xenium(
    path: Union[str, os.PathLike, Path],
    pixel_size: Number = 1,
    mode: Literal["dataframe", "mask"] = "mask"
    ) -> BoundariesData:
    # # read boundaries data
    path = Path(path)
    
    # create boundariesdata object
    #boundaries = BoundariesData(pixel_size=pixel_size)
    boundaries = BoundariesData()
    
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
                        
    else:
        cells_zarr_file = path / "cells.zarr.zip"
        
        # open zarr directory using dask
        data_dict = {
            "nuclear": dask.array.from_zarr(cells_zarr_file, component="masks/0"),
            "cellular": dask.array.from_zarr(cells_zarr_file, component="masks/1")
        }
    
    boundaries.add_boundaries(data=data_dict, pixel_size=pixel_size)

    return boundaries


def _read_matrix_from_xenium(path) -> AnnData:
    # extract parameters from metadata
    #cf_zarr_path = path / metadata["xenium_explorer_files"]["cell_features_zarr_filepath"]
    cf_h5_path = path / "cell_feature_matrix.h5"

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        # read matrix data
        adata = sc.read_10x_h5(cf_h5_path)

    # read cell information
    #cells_zarr_path = path / metadata["xenium_explorer_files"]["cells_zarr_filepath"]
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

def _restructure_transcripts_dataframe(dataframe):
    
    # decode columns
    dataframe = dataframe.apply(lambda x: decode_robust_series(x), axis=0)
    # set index and rename columns
    dataframe = dataframe.set_index("transcript_id")
    dataframe = dataframe.rename({
        "cell_id": "xenium_cell_id",
        "x_location": "x",
        "y_location": "y",
        "z_location": "z",
        "feature_name": "gene"
    }, axis=1)

    # reorder dataframe
    dataframe = dataframe.loc[:, ["x", "y", "z", "gene", "qv", "overlaps_nucleus", "fov_name", "nucleus_distance", "xenium_cell_id"]]

    # group column names into MultiIndices
    grouped_column_names = [
        ("coordinates", "x"),
        ("coordinates", "y"),
        ("coordinates", "z"),
        ("properties", "gene"),
        ("properties", "qv"),
        ("properties", "overlaps_nucleus"),
        ("properties", "fov_name"),
        ("properties", "nucleus_distance"),
        ("cell_id", "xenium")
    ]
    dataframe.columns = pd.MultiIndex.from_tuples(grouped_column_names)
    return dataframe
    

def read_celldata(
    path: Union[str, os.PathLike, Path],
    pixel_size: Number
    ) -> CellData:
    # read metadata
    path = Path(path)
    celldata_metadata = read_json(path / ".celldata")
    
    # read matrix data
    matrix = sc.read(path / celldata_metadata["matrix"])
    
    # read boundaries data
    # labels = convert_to_list(celldata_metadata["boundaries"].keys())
    # files = [path / f for f in convert_to_list(celldata_metadata["boundaries"].values())]
    boundaries_dict = {k: path / v for k,v in celldata_metadata["boundaries"].items()}
    boundaries_dict = {}
    for k,v in celldata_metadata["boundaries"].items():
        suffix = v.split(".", 1)[-1] # necessary to do this with split because of the two dots in .zarr.zip
        f = path / v
        if suffix == "parquet":
            d = pd.read_parquet(f)
        elif suffix == "zarr.zip":
            d = dask.array.from_zarr(f)
        else:
            raise ValueError(f"Boundaries saved in CellData object are neither .parquet nor .zarr.zip format: {suffix}")
        boundaries_dict[k] = d
    
    boundaries = BoundariesData()
    boundaries.add_boundaries(data=boundaries_dict, pixel_size=pixel_size)
    #boundaries.read_boundaries(files=files, labels=labels)
    
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
    metadata = read_json(path / "metadata.json")
    keys = metadata.keys()
    files = [path / f"{k}.geojson" for k in keys]
    data = AnnotationsData(files, keys)
    
    # overwrite metadata
    data.metadata = metadata
    return data


class XeniumData:
    #TODO: Docstring of XeniumData
    
    def __init__(self, 
                 path: Union[str, os.PathLike, Path],
                 pattern_xenium_folder: str = "output-{ins_id}__{slide_id}__{sample_id}",
                 metadata_filename: Optional[str] = None,
                 matrix: Optional[AnnData] = None
                 ):
        """_summary_

        Args:
            path (Union[str, os.PathLike, Path]): _description_
            pattern_xenium_folder (str, optional): _description_. Defaults to "output-{ins_id}__{slide_id}__{sample_id}".
            matrix (Optional[AnnData], optional): _description_. Defaults to None.

        Raises:
            FileNotFoundError: _description_
        """
        path = Path(path) # make sure the path is a pathlib path
        self.path = Path(path)
        self.dim = None # dimensions of the dataset
        self.from_xeniumdata = False  # flag indicating from where the data is read
        self.metadata_filename = ".xeniumdata"
        self.xd_metadata_file = self.path / self.metadata_filename
        if (self.xd_metadata_file).exists():
            # read xeniumdata metadata
            self.xd_metadata = read_json(self.xd_metadata_file)
            
            # read general xenium metadata
            self.metadata = read_json(self.path / "xenium.json")
            
            # retrieve slide_id and sample_id
            self.slide_id = self.xd_metadata["slide_id"]
            self.sample_id = self.xd_metadata["sample_id"]
            
            # set flag for xeniumdata
            self.from_xeniumdata = True
            
        elif matrix is None:
            # check if path exists
            if not self.path.is_dir():
                raise FileNotFoundError(f"No such directory found: {str(self.path)}")
            
            if metadata_filename is not None:
                self.metadata_filename = metadata_filename
                
            else:
                # check for modified metadata_filename
                metadata_files = [elem.name for elem in self.path.glob("*.xenium")]
                if "experiment_modified.xenium" in metadata_files:
                    self.metadata_filename = "experiment_modified.xenium"
                else:
                    self.metadata_filename = "experiment.xenium"
                
            # all changes are saved to the modified .xenium json
            self.metadata_save_path = self.path / "experiment_modified.xenium"
                
            # read metadata
            self.metadata = read_json(self.path / self.metadata_filename)
            
            # get slide id and sample id from metadata
            self.slide_id = self.metadata["slide_id"]
            self.sample_id = self.metadata["region_name"]
        else:
            self.cells.matrix = matrix
            self.slide_id = ""
            self.sample_id = ""
            self.path = Path("unknown/unknown")
            self.metadata_filename = ""
        
    def __repr__(self):
        repr = (
            f"{tf.Bold+tf.Red}XeniumData{tf.ResetAll}\n"
            f"{tf.Bold}Slide ID:{tf.ResetAll}\t{self.slide_id}\n"
            f"{tf.Bold}Sample ID:{tf.ResetAll}\t{self.sample_id}\n"
            f"{tf.Bold}Data path:{tf.ResetAll}\t{self.path.parent}\n"
            f"{tf.Bold}Data folder:{tf.ResetAll}\t{self.path.name}\n"
            f"{tf.Bold}Metadata file:{tf.ResetAll}\t{self.metadata_filename}"            
        )
        
        if hasattr(self, "images"):
            images_repr = self.images.__repr__()
            repr = (
                repr + f"\n{tf.SPACER+tf.RARROWHEAD} " + images_repr.replace("\n", f"\n{tf.SPACER}   ")
            )
                        
        if hasattr(self, "cells"):
            cells_repr = self.cells.__repr__()
            repr = (
                repr + f"\n{tf.SPACER+tf.RARROWHEAD+tf.Green+tf.Bold} cells{tf.ResetAll}\n{tf.SPACER}   " + cells_repr.replace("\n", f"\n{tf.SPACER}   ")
            )
        
        if hasattr(self, "transcripts"):
            trans_repr = f"DataFrame with shape {self.transcripts.shape[0]} x {self.transcripts.shape[1]}"
            
            repr = (
                repr + f"\n{tf.SPACER+tf.RARROWHEAD+tf.Purple+tf.Bold} transcripts{tf.ResetAll}\n{tf.SPACER}   " + trans_repr
            )
            
        if hasattr(self, "annotations"):
            annot_repr = self.annotations.__repr__()
            repr = (
                repr + f"\n{tf.SPACER+tf.RARROWHEAD} " + annot_repr.replace("\n", f"\n{tf.SPACER}   ")
            )
    
        if hasattr(self, "regions"):
            region_repr = self.regions.__repr__()
            repr = (
                repr + f"\n{tf.SPACER+tf.RARROWHEAD} " + region_repr.replace("\n", f"\n{tf.SPACER}   ")
            )
            
        if hasattr(self, "alt"):
            cells_repr = self.alt.__repr__()
            altseg_keys = self.alt.keys()
            repr = (
                #repr + f"\n{tf.SPACER+tf.RARROWHEAD+tf.Green+tf.Bold} alt{tf.ResetAll}\n{tf.SPACER}   " + cells_repr.replace("\n", f"\n{tf.SPACER}   ")
                repr + f"\n{tf.SPACER+tf.RARROWHEAD+tf.Green+tf.Bold} alt{tf.ResetAll}\n"
                f"{tf.SPACER}   Alternative CellData objects with following keys: {','.join(altseg_keys)}"
            )
        return repr

    def _save_metadata(self,
                      metadata_path: Union[str, os.PathLike, Path] = None
                      ):
        # if there is no specific path given, the metadata is written to the default path for modified metadata
        if metadata_path is None:
            metadata_path = self.metadata_save_path
            
        # write to json file
        metadata_json = json.dumps(self.metadata, indent=4)
        print(f"\t\tSave metadata to {metadata_path}", flush=True)
        with open(metadata_path, "w") as metafile:
            metafile.write(metadata_json)

    def assign_annotations(self, 
                annotation_key: str = "all",
                add_annotation_masks: bool = False
                ):
        '''
        Function to assign the annotations to the anndata object in XeniumData.matrix.
        Annotation information is added to the DataFrame in `.obs`.
        '''
        # assert that prerequisites are met
        assert hasattr(self, "cells"), "No .cells attribute available. Run `read_cells()`."
        assert hasattr(self, "annotations"), "No .annotations attribute available. Run `read_annotations()`."
        
        if annotation_key == "all":
            annotation_key = self.annotations.metadata.keys()
            
        # make sure annotation keys are a list
        annotation_key = convert_to_list(annotation_key)
        
        # convert coordinates into shapely Point objects
        points = [Point(elem) for elem in self.cells.matrix.obsm["spatial"]]

        # iterate through annotation keys
        for annotation_key in annotation_key:
            print(f"Assigning key '{annotation_key}'...")
            # extract pandas dataframe of current key
            annot = getattr(self.annotations, annotation_key)
            
            # get unique list of annotation names
            annot_names = annot.name.unique()
            
            # initiate dataframe as dictionary
            df = {}

            # iterate through names
            for n in annot_names:
                polygons = annot[annot.name == n].geometry.tolist()
                
                in_poly = []
                for poly in polygons:
                    # check if which of the points are inside the current annotation polygon
                    in_poly.append(poly.contains(points))
                
                # check if points were in any of the polygons
                in_poly_res = np.array(in_poly).any(axis=0)
                
                # collect results
                df[n] = in_poly_res
                
            # convert into pandas dataframe
            df = pd.DataFrame(df)
            df.index = self.cells.matrix.obs_names
            
            # create annotation from annotation masks
            df[f"annotation-{annotation_key}"] = [" & ".join(annot_names[row.values]) if np.any(row.values) else np.nan for i, row in df.iterrows()]
            
            if add_annotation_masks:
                self.cells.matrix.obs = pd.merge(left=self.cells.matrix.obs, right=df, left_index=True, right_index=True)
            else:
                self.cells.matrix.obs = pd.merge(left=self.cells.matrix.obs, right=df.iloc[:, -1], left_index=True, right_index=True)
                
            # save that the current key was analyzed
            self.annotations.metadata[annotation_key]["analyzed"] = tf.TICK
        
    def copy(self):
        '''
        Function to generate a deep copy of the XeniumData object.
        '''
        from copy import deepcopy
        had_viewer = False
        if hasattr(self, "viewer"):
            had_viewer = True
            
            # make copy of viewer to add it later again
            viewer_copy = self.viewer.copy()
            # remove viewer because there is otherwise a error during deepcopy
            del self.viewer
        
        # make copy
        self_copy = deepcopy(self)
        
        # add viewer again to original object if necessary
        if had_viewer:
            self.viewer = viewer_copy
        
        return self_copy
    
    def crop(self, 
            shape_layer: Optional[str] = None,
            xlim: Optional[Tuple[int, int]] = None,
            ylim: Optional[Tuple[int, int]] = None,
            inplace: bool = False
            ):
        '''
        Function to crop the XeniumData object.
        '''
        if shape_layer is not None:
            try:
                # extract shape layer for cropping from napari viewer
                crop_shape = self.viewer.layers[shape_layer]
            except KeyError:
                raise KeyError(f"Shape layer selected for cropping ('{shape_layer}') was not found in layers.")
            
            # check the type of the element
            if not isinstance(crop_shape, napari.layers.Shapes):
                raise WrongNapariLayerTypeError(found=type(crop_shape), wanted=napari.layers.Shapes)
            
            use_shape = True
        else:
            # if xlim or ylim is not none, assert that both are not None
            if xlim is not None or ylim is not None:
                assert np.all([elem is not None for elem in [xlim, ylim]])
                use_shape = False
            
        # assert that either shape_layer is given or xlim/ylim
        assert np.any([elem is not None for elem in [shape_layer, xlim, ylim]]), "No values given for either `shape_layer` or `xlim/ylim`."
        
        if use_shape:
            # extract shape layer for cropping from napari viewer
            crop_shape = self.viewer.layers[shape_layer]
            
            # check the structure of the shape object
            if len(crop_shape.data) != 1:
                raise NotOneElementError(crop_shape.data)
            
            # select the shape from list
            crop_window = crop_shape.data[0].copy()
            crop_window *= self.metadata["pixel_size"] # convert to metric unit (normally µm)
            
            # extract x and y limits from the shape (assuming a rectangle)
            xlim = (crop_window[:, 1].min(), crop_window[:, 1].max())
            ylim = (crop_window[:, 0].min(), crop_window[:, 0].max())
            
        # check if the changes are supposed to be made in place or not
        if inplace:
            _self = self
        else:
            _self = self.copy()
            
        # if the object was previously cropped, check if the current window is identical with the previous one
        if np.all([elem in _self.metadata.keys() for elem in ["cropping_xlim", "cropping_ylim"]]):
            # test whether the limits are identical
            if (xlim == _self.metadata["cropping_xlim"]) & (ylim == _self.metadata["cropping_ylim"]):
                raise XeniumDataRepeatedCropError(xlim, ylim)
        
        try:
            # infer mask from cell coordinates
            cells = _self.cells
        except AttributeError:
            pass
        else:
            cell_coords = cells.matrix.obsm['spatial'].copy()
            xmask = (cell_coords[:, 0] >= xlim[0]) & (cell_coords[:, 0] <= xlim[1])
            ymask = (cell_coords[:, 1] >= ylim[0]) & (cell_coords[:, 1] <= ylim[1])
            mask = xmask & ymask
            
            # select
            _self.cells.matrix = _self.cells.matrix[mask, :].copy()
            
            # crop boundaries
            _self.cells.boundaries.crop(
                cell_ids=_self.cells.matrix.obs_names, xlim=xlim, ylim=ylim
                )
            
            # sync cell ids of boundaries with matrix
            #_self.cells.sync_cell_ids()
            
            # shift coordinates to correct for change of coordinates during cropping
            _self.cells.shift(x=-xlim[0], y=-ylim[0])
            
        if hasattr(_self, "transcripts"):
            # infer mask for selection
            # xmask = (_self.transcripts["x_location"] >= xlim[0]) & (_self.transcripts["x_location"] <= xlim[1])
            # ymask = (_self.transcripts["y_location"] >= ylim[0]) & (_self.transcripts["y_location"] <= ylim[1])
            xmask = (_self.transcripts["coordinates", "x"] >= xlim[0]) & (_self.transcripts["coordinates", "x"] <= xlim[1])
            ymask = (_self.transcripts["coordinates", "y"] >= ylim[0]) & (_self.transcripts["coordinates", "y"] <= ylim[1])
            mask = xmask & ymask
            
            # select
            _self.transcripts = _self.transcripts.loc[mask, :].copy()
            
            # move origin again to 0 by subtracting the lower limits from the coordinates
            # _self.transcripts["x_location"] -= xlim[0]
            # _self.transcripts["y_location"] -= ylim[0]
            _self.transcripts["coordinates", "x"] -= xlim[0]
            _self.transcripts["coordinates", "y"] -= ylim[0]
            
        if hasattr(_self, "images"):
            _self.images.crop(xlim=xlim, ylim=ylim)
        
        if hasattr(_self, "annotations"):
            _self.annotations.crop(xlim=xlim, ylim=ylim)
            
        if hasattr(_self, "regions"):
            _self.regions.crop(xlim=xlim, ylim=ylim)
                
        # add information about cropping to metadata
        _self.metadata["cropping_xlim"] = xlim
        _self.metadata["cropping_ylim"] = ylim
                
        
        if inplace:
            if hasattr(self, "viewer"):
                del _self.viewer # delete viewer
        else:
            return _self

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

        sc.pp.highly_variable_genes(self.cells.matrix, batch_key=hvg_batch_key, flavor=hvg_flavor, layer=hvg_layer, n_top_genes=hvg_n_top_genes)


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
        check_raw(self.cells.matrix.X)

        # store raw counts in layer
        print("Store raw counts in anndata.layers['counts']...") if verbose else None
        self.cells.matrix.layers['counts'] = self.cells.matrix.X.copy()

        # preprocessing according to napari tutorial in squidpy
        print(f"Normalization, {transformation_method}-transformation...") if verbose else None
        sc.pp.normalize_total(self.cells.matrix)
        self.cells.matrix.layers['norm_counts'] = self.cells.matrix.X.copy()
        
        # transform either using log transformation or square root transformation
        if transformation_method == "log1p":
            sc.pp.log1p(self.cells.matrix)
        elif transformation_method == "sqrt":
            # Suggested in stlearn tutorial (https://stlearn.readthedocs.io/en/latest/tutorials/Xenium_PSTS.html)
            X = self.cells.matrix.X.toarray()   
            self.cells.matrix.X = csr_matrix(np.sqrt(X) + np.sqrt(X + 1))
        else:
            raise ValueError(f'`transformation_method` is not one of ["log1p", "sqrt"]')
        


    def plot_dimred(self, save: Optional[str] = None):
        '''
        Read dimensionality reduction plots.
        '''
        # construct paths
        analysis_path = self.path / "analysis"
        umap_file = analysis_path / "umap" / "gene_expression_2_components" / "projection.csv"
        pca_file = analysis_path / "pca" / "gene_expression_10_components" / "projection.csv"
        cluster_file = analysis_path / "clustering" / "gene_expression_graphclust" / "clusters.csv"
        
        
        # read data
        umap_data = pd.read_csv(umap_file)
        pca_data = pd.read_csv(pca_file)
        cluster_data = pd.read_csv(cluster_file)
        
        # merge dimred data with clustering data
        data = ft.reduce(lambda left, right: pd.merge(left, right, on='Barcode'), [umap_data, pca_data.iloc[:, :3], cluster_data])
        data["Cluster"] = data["Cluster"].astype('category')
                        
        # plot
        nrows = 1
        ncols = 2
        fig, axs = plt.subplots(nrows, ncols, figsize=(8*ncols, 6*nrows))
        sns.scatterplot(data=data, x="PC-1", y="PC-2", hue="Cluster", palette="tab20", ax=axs[0])
        sns.scatterplot(data=data, x="UMAP-1", y="UMAP-2", hue="Cluster", palette="tab20", ax=axs[1])
        if save is not None:
            plt.savefig(save)
        plt.show()

    def read_all(self, 
                 skip: Optional[str] = None,
                 ):
        # extract read functions
        read_funcs = [elem for elem in dir(self) if elem.startswith("read_")]
        read_funcs = [elem for elem in read_funcs if elem != "read_all"]
        
        for f in read_funcs:
            if skip is None or skip not in f:
                func = getattr(self, f)
                try:
                    func()
                except ModalityNotFoundError as err:
                    print(err)

    def read_annotations(self,
                    annotations_dir: Union[str, os.PathLike, Path] = None, # "../annotations",
                    suffix: str = ".geojson",
                    pattern_annotations_file: str = "annotations-{slide_id}__{sample_id}__{name}"
                    ):
        print("Reading annotations...", flush=True)
        if self.from_xeniumdata:
            try:
                p = self.xd_metadata["annotations"]
            except KeyError:
                raise ModalityNotFoundError(modality="annotations")
            self.annotations = read_annotationsdata(path=self.path / p)
            
            # # check if annotations data is stored in this XeniumData
            # if "annotations" not in self.xd_metadata:
            #     raise ModalityNotFoundError(modality="annotations")
            
            # # get path and names of annotation files
            # keys = self.xd_metadata["annotations"].keys()
            # files = [self.path / self.xd_metadata["annotations"][n] for n in keys]
            
        else:
            if annotations_dir is None:
                raise ModalityNotFoundError(modality="annotations")
            else:
                # convert to Path
                annotations_dir = Path(annotations_dir)
                
                # check if the annotation path exists. If it does not, first assume that it is a relative path and check that.
                if not annotations_dir.is_dir():
                    annotations_dir = Path(os.path.normpath(self.path / annotations_dir))
                    if not annotations_dir.is_dir():
                        raise FileNotFoundError(f"`annotations_dir` {annotations_dir} is neither a direct path nor a relative path.")
                
                # get list annotation files that match the current slide id and sample id
                files = []
                keys = []
                for file in annotations_dir.glob(f"*{suffix}"):
                    if self.slide_id in str(file.stem) and (self.sample_id in str(file.stem)):
                        parsed = parse(pattern_annotations_file, file.stem)
                        keys.append(parsed.named["name"])
                        files.append(file)
            
            self.annotations = AnnotationsData(files=files, keys=keys, pixel_size=self.metadata['pixel_size'])
            
    def parse_baysor(self, 
                    baysor_output: Union[str, os.PathLike, Path],
                    key_to_add: str = "baysor",
                    pixel_size: Number = 1 # the pixel size is usually 1 since baysor runs on the µm coordinates
                    ):
        
        try:
            cells = self.cells
        except AttributeError:
            raise ModalityNotFoundError(modality="cells")
        
        # read matrix
        print("Parsing count matrix...", flush=True)
        loomfile = baysor_output / "segmentation_counts.loom"
        matrix = sc.read_loom(loomfile)
        
        # read polygons
        print("Reading segmentation masks", flush=True)
        print("\tRead polygons", flush=True)
        jsonfile = baysor_output / "segmentation_polygons.json"
        df = read_baysor_polygons(jsonfile)
        
        # determine dimensions of dataset
        xmax = ceil(cells.matrix.obsm['spatial'][:, 0].max() + 15)
        ymax = ceil(cells.matrix.obsm['spatial'][:, 1].max() + 15)
        
        # generate a segmentation mask
        print("\tConvert polygons to segmentation mask", flush=True)
        img = rasterize(list(zip(df["geometry"], range(1, len(df)+1))), out_shape=(ymax,xmax))
        
        # convert to dask array
        img = da.from_array(img)
        
        # create boundaries object
        boundaries = BoundariesData()
        boundaries.add_boundaries(data={key_to_add: img}, pixel_size=pixel_size)
        
        # add data to XeniumData
        alt_attr_name = "alt"
        try:
            alt_attr = getattr(self, alt_attr_name)
        except AttributeError:
            setattr(self, alt_attr_name, {})
            alt_attr = getattr(self, alt_attr_name)
        
        alt_attr[key_to_add] = CellData(matrix=matrix, boundaries=boundaries)
        
        trans_attr_name = "transcripts"
        try:
            trans_attr = getattr(self, trans_attr_name)
        except AttributeError:
            print("No transcript layer found. Addition of Baysor transcript data is skipped.", flush=True)
            pass
        else:
            # read transcripts from Baysor results
            print("Parsing transcripts data...", flush=True)
            
            print("\tRead data", flush=True)
            segcsv_file = baysor_output / "segmentation.csv"
            baysor_transcript_dataframe = pd.read_csv(segcsv_file)
            
            print("\tMerge with existing data", flush=True)
            baysor_results = baysor_transcript_dataframe.set_index("transcript_id")[["cell"]]
            baysor_results.columns = pd.MultiIndex.from_tuples([("cell_id", key_to_add)])
            trans_attr = pd.merge(left=trans_attr,
                                  right=baysor_results, 
                                  left_index=True, 
                                  right_index=True
                                  )
            
            # add resulting dataframe to XeniumData
            setattr(self, trans_attr_name, trans_attr)

        
    def read_regions(self,
                    regions_dir: Union[str, os.PathLike, Path] = None, # "../regions",
                    suffix: str = ".geojson",
                    pattern_regions_file: str = "regions-{slide_id}__{sample_id}__{name}"
                    ):
        print("Reading regions...", flush=True)
        if self.from_xeniumdata:
            try:
                p = self.xd_metadata["regions"]
            except KeyError:
                raise ModalityNotFoundError(modality="regions")
            self.regions = read_regionsdata(path=self.path / p)
            
        else:
            if regions_dir is None:
                raise ModalityNotFoundError(modality="regions")
            else:
                # convert to Path
                regions_dir = Path(regions_dir)
                
                # check if the annotation path exists. If it does not, first assume that it is a relative path and check that.
                if not regions_dir.is_dir():
                    regions_dir = Path(os.path.normpath(self.path / regions_dir))
                    if not regions_dir.is_dir():
                        raise FileNotFoundError(f"`regions_dir` {regions_dir} is neither a direct path nor a relative path.")
                
                # get list annotation files that match the current slide id and sample id
                files = []
                keys = []
                for file in regions_dir.glob(f"*{suffix}"):
                    if self.slide_id in str(file.stem) and (self.sample_id in str(file.stem)):
                        parsed = parse(pattern_regions_file, file.stem)
                        keys.append(parsed.named["name"])
                        files.append(file)
            
            self.regions = RegionsData(files=files, keys=keys, pixel_size=self.metadata['pixel_size'])


    def read_cells(self):
        print("Reading cells...", flush=True)
        pixel_size = self.metadata["pixel_size"]
        if self.from_xeniumdata:
            try:
                cells_path = self.xd_metadata["cells"]
            except KeyError:
                raise ModalityNotFoundError(modality="cells")
            self.cells = read_celldata(path=self.path / cells_path, pixel_size=pixel_size)
        else:
            # read celldata
            matrix = matrix = _read_matrix_from_xenium(path=self.path)
            boundaries = _read_boundaries_from_xenium(path=self.path, pixel_size=pixel_size)
            self.cells = CellData(matrix=matrix, boundaries=boundaries)
            
            # read binned expression
            arr = _read_binned_expression(path=self.path, gene_names_to_select=self.cells.matrix.var_names)
            self.cells.matrix.varm["binned_expression"] = arr
            

    def read_images(self,
                    names: Union[Literal["all", "nuclei"], str] = "all", # here a specific image can be chosen
                    nuclei_type: Literal["focus", "mip", ""] = "mip"
                    ):
        if self.from_xeniumdata:
            # check if matrix data is stored in this XeniumData
            if "images" not in self.xd_metadata:
                raise ModalityNotFoundError(modality="images")
            
            # get file paths and names
            img_files = list(self.xd_metadata["images"].values())
            img_names = list(self.xd_metadata["images"].keys())
        else:
            if names == "nuclei":
                img_keys = [f"morphology_{nuclei_type}_filepath"]
                img_names = ["nuclei"]
            else:
                # get available keys for registered images in metadata
                img_keys = [elem for elem in self.metadata["images"] if elem.startswith("registered")]
                
                # extract image names from keys and add nuclei
                img_names = ["nuclei"] + [elem.split("_")[1] for elem in img_keys]
                
                # add dapi image key
                img_keys = [f"morphology_{nuclei_type}_filepath"] + img_keys
                
                if names != "all":
                    # make sure keys is a list
                    names = convert_to_list(names)
                    # select the specified keys
                    mask = [elem in names for elem in img_names]
                    img_keys = [elem for m, elem in zip(mask, img_keys) if m]
                    img_names = [elem for m, elem in zip(mask, img_names) if m]
                    
            # get path of image files
            img_files = [self.metadata["images"][k] for k in img_keys]
            
        # load image into ImageData object
        print("Reading images...", flush=True)
        self.images = ImageData(self.path, img_files, img_names, pixel_size=self.metadata['pixel_size'])

    def read_transcripts(self,
                        transcript_filename: str = "transcripts.parquet"
                        ):
        if self.from_xeniumdata:
            # check if matrix data is stored in this XeniumData
            if "transcripts" not in self.xd_metadata:
                raise ModalityNotFoundError(modality="transcripts")
            
            # read transcripts
            print("Reading transcripts...", flush=True)
            self.transcripts = pd.read_parquet(self.path / self.xd_metadata["transcripts"])
        else:
            # read transcripts
            print("Reading transcripts...", flush=True)
            transcript_dataframe = pd.read_parquet(self.path / transcript_filename)
            
            self.transcripts = _restructure_transcripts_dataframe(transcript_dataframe)
            

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
            sc.pp.pca(self.cells.matrix)
            if umap:
                sc.pp.neighbors(self.cells.matrix)
                sc.tl.umap(self.cells.matrix)
            if tsne:
                sc.tl.tsne(self.cells.matrix, n_jobs=tsne_jobs, learning_rate=tsne_lr)

        else:
            # PCA
            sc.pp.pca(self.cells.matrix)

            neigh_uncorr_key = 'neighbors_uncorrected'
            sc.pp.neighbors(self.cells.matrix, key_added=neigh_uncorr_key)

            # clustering
            sc.tl.leiden(self.cells.matrix, neighbors_key=neigh_uncorr_key, key_added='leiden_uncorrected')  

            # batch correction
            print(f"Batch correction using scanorama for {batch_correction_key}...") if verbose else None
            hvgs = list(self.cells.matrix.var_names[self.cells.matrix.var['highly_variable']])
            self.cells.matrix = scanorama(self.cells.matrix, batch_key=batch_correction_key, hvg=hvgs, verbose=False, **kwargs)

            # find neighbors
            sc.pp.neighbors(self.cells.matrix, use_rep="X_scanorama")
            sc.tl.umap(self.cells.matrix)
            sc.tl.tsne(self.cells.matrix, use_rep="X_scanorama")

        # clustering
        print("Leiden clustering...") if verbose else None
        sc.tl.leiden(self.cells.matrix)


    def register_images(self,
                        img_dir: Union[str, os.PathLike, Path],
                        img_suffix: str = ".ome.tif",
                        pattern_img_file: str = "{slide_id}__{sample_id}__{image_names}__{image_type}",
                        decon_scale_factor: float = 0.2,
                        image_name_sep: str = "_",  # string separating the image names in the file name
                        nuclei_name: str = "DAPI",  # name used for the nuclei image
                        physicalsize: str = 'µm',
                        #dapi_channel: int = None
                        ):
        '''
        Register images stored in XeniumData object.
        '''

        # add arguments to object
        self.img_dir = Path(img_dir)
        self.pattern_img_file = pattern_img_file
        
        # check if image path exists
        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"No such directory found: {str(self.img_dir)}")
        
        print(f"Processing sample {tf.Bold}{self.sample_id}{tf.ResetAll} of slide {tf.Bold}{self.slide_id}{tf.ResetAll}", flush=True)        
        
        # get a list of image files
        img_files = sorted(self.img_dir.glob("*{}".format(img_suffix)))
        
        # find the corresponding image
        corr_img_files = [elem for elem in img_files if self.slide_id in str(elem) and self.sample_id in str(elem)]
        
        # make sure images corresponding to the Xenium data were found
        if len(corr_img_files) == 0:
            print(f'\tNo image corresponding to slide `{self.slide_id}` and sample `{self.sample_id}` were found.')
        else:
            if self.metadata_filename == "experiment_modified.xenium":
                print(f"\tFound modified `{self.metadata_filename}` file. Information will be added to this file.")
            elif self.metadata_filename == "experiment.xenium":
                print(f"\tOnly unmodified metadata file (`{self.metadata_filename}`) found. Information will be added to new file (`experiment_modified.xenium`).")
            else:
                raise FileNotFoundError("Metadata file not found.")

            for img_file in corr_img_files:
                # parse name of current image
                img_stem = img_file.stem.split(".")[0] # make sure to remove also suffices like .ome.tif
                img_file_parsed = parse(pattern_img_file, img_stem)
                self.image_names = img_file_parsed.named["image_names"].split(image_name_sep)
                image_type = img_file_parsed.named["image_type"] # check which image type it has (`histo` or `IF`)
                
                # determine the structure of the image axes and check other things
                axes_template = "YX"
                if image_type == "histo":
                    axes_image = "YXS"
                    
                    # make sure that there is only one image name given
                    if len(self.image_names) > 1:
                        raise ValueError(f"More than one image name retrieved ({self.image_names})")
                    
                    if len(self.image_names) == 0:
                        raise ValueError(f"No image name found in file {img_file}")
                    
                elif image_type == "IF":
                    axes_image = "CYX"
                else:
                    raise UnknownOptionError(image_type, available=["histo", "IF"])
                
                print(f'\tProcessing following {image_type} images: {tf.Bold}{", ".join(self.image_names)}{tf.ResetAll}', flush=True)

                # read images
                print("\t\tLoading images to be registered...", flush=True)
                image = dask_image.imread.imread(img_file) # e.g. HE image
                
                # sometimes images are read with an empty time dimension in the first axis. 
                # If this is the case, it is removed here.
                if len(image.shape) == 4:
                    image = image[0]
                    
                # read images in XeniumData object
                self.read_images(names="nuclei")
                template = self.images.nuclei[0] # usually the nuclei/DAPI image is the template. Use highest resolution of pyramid.
                
                # extract OME metadata
                ome_metadata_template = self.images.metadata["nuclei"]["OME"]
                # extract pixel size for x and y from metadata
                pixelsizes = {key: ome_metadata_template['Image']['Pixels'][key] for key in ['PhysicalSizeX', 'PhysicalSizeY']}
                
                # the selected image will be a grayscale image in both cases (nuclei image or deconvolved hematoxylin staining)
                axes_selected = "YX" 
                if image_type == "histo":
                    print("\t\tRun color deconvolution", flush=True)
                    # deconvolve HE - performed on resized image to save memory
                    # TODO: Scale to max width instead of using a fixed scale factor before deconvolution (`scale_to_max_width`)
                    nuclei_img, eo, dab = deconvolve_he(img=resize_image(image, scale_factor=decon_scale_factor, axes=axes_selected), 
                                                return_type="grayscale", convert=True)

                    # bring back to original size
                    nuclei_img = resize_image(nuclei_img, scale_factor=1/decon_scale_factor, axes=axes_selected)
                    
                    # set nuclei_channel and nuclei_axis to None
                    nuclei_channel = channel_axis = None
                else:
                    # image_type is "IF" then
                    # get index of nuclei channel
                    nuclei_channel = self.image_names.index(nuclei_name)
                    channel_axis = axes_image.find("C")
                    
                    if channel_axis == -1:
                        raise ValueError(f"No channel indicator `C` found in image axes ({axes_image})")
                    
                    print(f"\t\tSelect image with nuclei from IF image (channel: {nuclei_channel})", flush=True)
                    # select nuclei channel from IF image
                    if nuclei_channel is None:
                        raise TypeError("Argument `nuclei_channel` should be an integer and not NoneType.")
                    
                    # select dapi channel for registration
                    nuclei_img = np.take(image, nuclei_channel, channel_axis)
                    #selected = image[nuclei_channel]
                    
                # Setup image registration objects - is important to load and scale the images.
                # The reason for this are limits in C++, not allowing to perform certain OpenCV functions on big images.
                
                # First: Setup the ImageRegistration object for the whole image (before deconvolution in histo images and multi-channel in IF)
                imreg_complete = ImageRegistration(
                    image=image,
                    template=template,
                    axes_image=axes_image,
                    axes_template=axes_template,
                    verbose=False
                    )
                # load and scale the whole image
                imreg_complete.load_and_scale_images()

                # setup ImageRegistration object with the nucleus image (either from deconvolution or just selected from IF image)
                imreg_selected = ImageRegistration(
                    image=nuclei_img,
                    template=imreg_complete.template,
                    axes_image=axes_selected,
                    axes_template=axes_template,
                    max_width=4000,
                    convert_to_grayscale=False,
                    perspective_transform=False
                )
                
                # run all steps to extract features and get transformation matrix
                imreg_selected.load_and_scale_images()
                
                print("\t\tExtract common features from image and template", flush=True)
                # perform registration to extract the common features ptsA and ptsB
                imreg_selected.extract_features()
                imreg_selected.calculate_transformation_matrix()
                
                if image_type == "histo":
                    # in case of histo RGB images, the channels are in the third axis and OpenCV can transform them
                    if imreg_complete.image_resized is None:
                        imreg_selected.image = imreg_complete.image  # use original image
                    else:
                        imreg_selected.image_resized = imreg_complete.image_resized  # use resized original image
                    
                    # perform registration
                    imreg_selected.perform_registration()
                    
                    # generate OME metadata for saving
                    metadata = {
                        **{'SignificantBits': 8,
                        'PhysicalSizeXUnit': 'µm',
                        'PhysicalSizeYUnit': 'µm'
                        },
                        **pixelsizes
                    }
                        
                    # save files
                    imreg_selected.save(path=self.path,
                                        filename=f"{self.slide_id}__{self.sample_id}__{self.image_names[0]}",
                                        axes=axes_image,
                                        photometric='rgb',
                                        ome_metadata=metadata
                                        )
                    
                    # save metadata
                    self.metadata['images'][f'registered_{self.image_names[0]}_filepath'] = os.path.relpath(imreg_selected.outfile, self.path).replace("\\", "/")
                    self._save_metadata()
                        
                    del imreg_complete, imreg_selected, image, template, nuclei_img, eo, dab
                else:
                    # image_type is IF
                    # In case of IF images the channels are normally in the first axis and each channel is registered separately
                    # Further, each channel is then saved separately as grayscale image.
                    
                    # iterate over channels
                    for i, n in enumerate(self.image_names):
                        # skip the DAPI image
                        if n == nuclei_name:
                            break
                        
                        if imreg_complete.image_resized is None:
                            # select one channel from non-resized original image
                            imreg_selected.image = np.take(imreg_complete.image, i, channel_axis)
                        else:
                            # select one channel from resized original image
                            imreg_selected.image_resized = np.take(imreg_complete.image_resized, i, channel_axis)
                            
                        # perform registration
                        imreg_selected.perform_registration()
                        
                        # save files
                        imreg_selected.save(path=self.path,
                                        filename=f"{self.slide_id}__{self.sample_id}__{n}",
                                        axes='YX',
                                        photometric='minisblack'
                                        )
                        
                        # save metadata
                        self.metadata['images'][f'registered_{n}_filepath'] = os.path.relpath(imreg_selected.outfile, self.path)
                        self._save_metadata()

                    # free RAM
                    del imreg_complete, imreg_selected, image, template, nuclei_img
                gc.collect()

        # read images
        self.read_images()

    def save(self,
            path: Union[str, os.PathLike, Path],
            overwrite: bool = False,
            #zip: bool = False
            ):
        '''
        Function to save the XeniumData object.
        
        Args:
            path: Path to save the data to.
        '''
        # check if the path already exists
        # TODO: check the logic of the "zip part" below. Maybe it makes more sense to infer zip/no zip from path name?
        path = Path(path)
        
        # check overwrite
        check_overwrite_and_remove_if_true(path=path, overwrite=overwrite)
        
        # check whether to save to zip
        zip_output = check_zip(path=path)
        
        # remove zip if available
        path_stem = path.parent / path.stem
        
        if zip_output:
            check_overwrite_and_remove_if_true(path=path_stem, overwrite=overwrite)

        # create output directory if it does not exist yet
        path_stem.mkdir(parents=True, exist_ok=True)
        
        # create a metadata dictionary
        metadata = {}
        
        # store basic information about experiment
        metadata["slide_id"] = self.slide_id
        metadata["sample_id"] = self.sample_id
        #metadata["path"] = str(abspath(self.path))
        
        # save images
        try:
            images = self.images
        except AttributeError:
            pass
        else:
            _save_images(
                imagedata=images,
                path=path_stem,
                metadata=metadata,
                images_as_zarr=True
                )

        # save cells
        try:
            cells = self.cells
        except AttributeError:
            pass
        else:
            _save_cells(
                cells=cells,
                path=path_stem,
                metadata=metadata
            )
            
        # save alternative cell data
        try:
            alt = self.alt
        except AttributeError:
            pass
        else:
            _save_alt(
                attr=alt,
                path=path_stem,
                metadata=metadata
            )

            
        # save transcripts
        try:
            transcripts = self.transcripts
        except AttributeError:
            pass
        else:
            _save_transcripts(
                transcripts=transcripts,
                path=path_stem,
                metadata=metadata
                )
                
        
        # save annotations
        try:
            annotations = self.annotations
        except AttributeError:
            pass
        else:
            _save_annotations(
                annotations=annotations,
                path=path_stem,
                metadata=metadata
            )
            
            
        # save regions
        try:
            regions = self.regions
        except AttributeError:
            pass
        else:
            _save_regions(
                regions=regions,
                path=path_stem,
                metadata=metadata
            )

        # save version of InSituPy
        metadata["version"] = __version__
            
        # write Xeniumdata metadata to json file
        xd_metadata_path = path_stem / ".xeniumdata"
        write_dict_to_json(dictionary=metadata, file=xd_metadata_path)
            
        # write Xenium metadata to json file
        metadata_path = path_stem / "xenium.json"
        write_dict_to_json(dictionary=self.metadata, file=metadata_path)
        
        # Optionally: zip the resulting directory
        if zip_output:
            shutil.make_archive(path_stem, 'zip', path_stem, verbose=False)
            shutil.rmtree(path_stem) # delete directory

    def quicksave(self):
        # create quicksave directory if it does not exist already
        quicksave_dir = CACHE / "quicksaves"
        quicksave_dir.mkdir(parents=True, exist_ok=True)
        
        # create filename
        current_datetime = datetime.now().strftime("%y%m%d_%H-%M-%S")
        slide_id = self.slide_id
        sample_id = self.sample_id
        uid = str(uuid4())[:8]

        # create output directory
        outname = f"{slide_id}__{sample_id}__{current_datetime}__{uid}"
        outdir = quicksave_dir / outname
        
        # save annotations
        try:
            annotations = self.annotations
        except AttributeError:
            pass
        else:
            _save_annotations(
                annotations=annotations,
                path=outdir,
                metadata=None
            )
            
        # # zip the output
        shutil.make_archive(outdir, format='zip', root_dir=outdir, verbose=False)
        shutil.rmtree(outdir) # delete directory
            
        
    def list_quicksaves(self):
        # create quicksave directory if it does not exist already
        quicksave_dir = CACHE / "quicksaves"
        
        pattern = "{slide_id}__{sample_id}__{savetime}__{uid}"

        # collect results
        res = {
            "slide_id": [],
            "sample_id": [],
            "savetime": [],
            "uid": []
        }
        for f in quicksave_dir.glob("*"):
            parse_res = parse(pattern, f.stem).named
            for key, value in parse_res.items():
                res[key].append(value)
            
        # create and return dataframe
        return pd.DataFrame(res)
            
            
    def show(self,
        keys: Optional[str] = None,
        annotation_keys: Optional[str] = None,
        show_images: bool = True,
        show_cells: bool = False,
        point_size: int = 6,
        scalebar: bool = True,
        pixel_size: float = None, # if none, extract from metadata
        unit: str = "µm",
        cmap_annotations: str ="Dark2",
        grayscale_colormap: List[str] = ["red", "green", "cyan", "magenta", "yellow", "gray"],
        return_viewer: bool = False,
        widgets_max_width: int = 200
        ):
        # get information about pixel size
        if (pixel_size is None) & (scalebar):
            # extract pixel_size
            pixel_size = float(self.metadata["pixel_size"])
        else:
            pixel_size = 1
        
        # create viewer
        self.viewer = napari.Viewer()
        
        # optionally add images
        if show_images:
            # add images
            if not hasattr(self, "images"):
                raise XeniumDataMissingObject("images")
                
            image_keys = self.images.metadata.keys()
            n_grayscales = 0 # number of grayscale images
            for i, img_name in enumerate(image_keys):
                img = getattr(self.images, img_name)
                is_visible = False if i < len(image_keys) - 1 else True # only last image is set visible
                
                # check if the current image is RGB
                is_rgb = self.images.metadata[img_name]["rgb"]
                
                if is_rgb:
                    cmap = None  # default value of cmap
                    blending = "translucent_no_depth"  # set blending mode
                else:
                    if img_name == "nuclei":
                        cmap = "blue"
                    else:
                        cmap = grayscale_colormap[n_grayscales]
                        n_grayscales += 1
                    blending = "additive"  # set blending mode
                
                # create image pyramid for lazy loading
                img_pyramid = create_img_pyramid(img=img, nsubres=6)
                                
                # add img pyramid to napari viewer
                self.viewer.add_image(
                        img_pyramid,
                        name=img_name,
                        colormap=cmap,
                        blending=blending,
                        rgb=is_rgb,
                        contrast_limits=self.images.metadata[img_name]["contrast_limits"],
                        scale=(pixel_size, pixel_size),
                        visible=is_visible
                    )
        
        # optionally: add cells as points
        if show_cells or keys is not None:
            try:
                cells = self.cells
            except AttributeError:
                raise XeniumDataMissingObject("cells")
            else:      
                # convert keys to list
                keys = convert_to_list(keys)
                
                # get point coordinates
                points = np.flip(cells.matrix.obsm["spatial"].copy(), axis=1) # switch x and y (napari uses [row,column])
                #points *= pixel_size # convert to length unit (e.g. µm)
            
                # get expression matrix
                if issparse(cells.matrix.X):
                    X = cells.matrix.X.toarray()
                else:
                    X = cells.matrix.X

                for i, k in enumerate(keys):
                    pvis = False if i < len(keys) - 1 else True # only last image is set visible
                    # get expression values
                    if k in cells.matrix.obs.columns:
                        color_value = cells.matrix.obs[k].values
                    
                    else:
                        geneid = cells.matrix.var_names.get_loc(k)
                        color_value = X[:, geneid]
                        
                    # extract names of cells
                    cell_names = cells.matrix.obs_names.values

                    # create points layer
                    layer = _create_points_layer(
                        points=points,
                        color_values=color_value,
                        name=k,
                        point_names=cell_names,
                        point_size=point_size,
                        visible=pvis
                    )
                    
                    # add layer programmatically - does not work for all types of layers
                    # see: https://forum.image.sc/t/add-layerdatatuple-to-napari-viewer-programmatically/69878
                    self.viewer.add_layer(Layer.create(*layer))            

        # optionally add annotations
        if annotation_keys is not None:        
            # get colorcycle for region annotations
            cmap_annot = matplotlib.colormaps[cmap_annotations]
            cc_annot = cmap_annot.colors
            
            if annotation_keys == "all":
                annotation_keys = self.annotations.metadata.keys()
            annotation_keys = convert_to_list(annotation_keys)
            for annotation_key in annotation_keys:
                annot_df = getattr(self.annotations, annotation_key)
                
                # get classes
                classes = annot_df['name'].unique()
                
                # iterate through classes
                for cl in classes:
                    # generate layer name
                    layer_name = f"*{cl} ({annotation_key})"
                    
                    # get dataframe for this class
                    class_df = annot_df[annot_df["name"] == cl]
                    
                    if layer_name not in self.viewer.layers:
                        # add layer to viewer
                        _add_annotations_as_layer(
                            dataframe=class_df,
                            viewer=self.viewer,
                            layer_name=layer_name
                        )
                    
        # WIDGETS
        try:
            cells = self.cells
        except AttributeError:
            pass
        else:
            # initialize the widgets
            add_points_widget, locate_cells_widget, add_region_widget, add_annotations_widget, add_boundaries_widget = _initialize_widgets(xdata=self)
            
            # add widgets to napari window
            if add_points_widget is not None:
                self.viewer.window.add_dock_widget(add_points_widget, name="Add cells", area="right")
                add_points_widget.max_height = 130
                add_points_widget.max_width = widgets_max_width
                
            if add_boundaries_widget is not None:
                self.viewer.window.add_dock_widget(add_boundaries_widget, name="Add boundaries", area="right")
                add_boundaries_widget.max_height = 80
                add_boundaries_widget.max_width = widgets_max_width
            
            if locate_cells_widget is not None:
                self.viewer.window.add_dock_widget(locate_cells_widget, name="Navigate", area="right")
                locate_cells_widget.max_height = 130
                locate_cells_widget.max_width = widgets_max_width
            
            if add_region_widget is not None:
                self.viewer.window.add_dock_widget(add_region_widget, name="Show regions", area="right")
                add_region_widget.max_height = 150
                add_region_widget.max_width = widgets_max_width
                
            if add_annotations_widget is not None:
                self.viewer.window.add_dock_widget(add_annotations_widget, name="Show annotations", area="right")
                add_annotations_widget.max_height = 150
                add_annotations_widget.max_width = widgets_max_width
        
        # add annotation widget to napari
        annot_widget = _annotation_widget()
        annot_widget.max_height = 100
        annot_widget.max_width = widgets_max_width
        self.viewer.window.add_dock_widget(annot_widget, name="Add annotations", area="right")
        
        # EVENTS
        # Function assign to an layer addition event
        def _update_uid(event):
            if event is not None:
                
                layer = event.source
                # print(event.action) # print what event.action returns
                # print(event.data_indices) # print index of event
                if event.action == "add":
                    # print(f'Added to {layer}')
                    if 'uid' in layer.properties:
                        layer.properties['uid'][-1] = str(uuid4())
                    else:
                        layer.properties['uid'] = np.array([str(uuid4())], dtype='object')
                    
                elif event.action == "remove":
                    pass
                    # print(f"Removed from {layer}")
                else:
                    raise ValueError("Unexpected value '{event.action}' for `event.action`. Expected 'add' or 'remove'.")

                # print(layer.properties)

        for layer in self.viewer.layers:
            if isinstance(layer, Shapes):
                layer.events.data.connect(_update_uid)
                #layer.metadata = layer.properties
                
        # Connect the function to all shapes layers in the viewer
        def connect_to_all_shapes_layers(event):
            layer = event.source[event.index]
            if event is not None and isinstance(layer, Shapes):
                # print('Annotation layer added')
                layer.events.data.connect(_update_uid)

        # Connect the function to any new layers added to the viewer
        self.viewer.layers.events.inserted.connect(connect_to_all_shapes_layers)
        
        # NAPARI SETTINGS
        if scalebar:
            # add scale bar
            self.viewer.scale_bar.visible = True
            self.viewer.scale_bar.unit = unit
        
        napari.run()
        if return_viewer:
            return self.viewer
    
    def store_annotations(self,
                        name_pattern = "*{class_name} ({annot_key})",
                        uid_col: str = "id"
                        ):
        '''
        Function to extract annotation layers from shapes layers and store them in the XeniumData object.
        '''
        try:
            viewer = self.viewer
        except AttributeError as e:
            print(f"{str(e)}. Use `.show()` first to open a napari viewer.")
            
        # iterate through layers and save them as annotation if they meet requirements
        layers = viewer.layers
        collection_dict = {}
        for layer in layers:
            if not isinstance(layer, Shapes):
                pass
            else:
                name_parsed = parse(name_pattern, layer.name)
                if name_parsed is not None:
                    annot_key = name_parsed.named["annot_key"]
                    class_name = name_parsed.named["class_name"]
                    
                    # if the XeniumData object does not has an annotations attribute, initialize it
                    if not hasattr(self, "annotations"):
                        self.annotations = AnnotationsData() # initialize empty object
                    
                    # extract shapes coordinates and colors
                    shapes = layer.data
                    colors = layer.edge_color.tolist()
                    
                    # scale coordinates
                    #shapes = [elem / self.metadata["pixel_size"] for elem in shapes]
                    
                    # build annotation GeoDataFrame
                    annot_df = {
                        uid_col: layer.properties["uid"],
                        "objectType": "annotation",
                        "geometry": [Polygon(np.stack([ar[:, 1], ar[:, 0]], axis=1)) for ar in shapes],  # switch x/y
                        "name": class_name,
                        "color": [[int(elem[e]*255) for e in range(3)] for elem in colors]
                    }
                    
                    # generate GeoDataFrame
                    annot_df = GeoDataFrame(annot_df, geometry="geometry")
                    
                    # add annotations
                    self.annotations.add_annotation(data=annot_df, key=annot_key, verbose=True)                       
 