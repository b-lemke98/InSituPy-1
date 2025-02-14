
import functools as ft
import gc
import json
import os
import shutil
from datetime import datetime
from numbers import Number
from os.path import abspath
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union
from uuid import uuid4
from warnings import warn

import anndata
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata._core.anndata import AnnData
from dask_image.imread import imread
from geopandas import GeoDataFrame
from parse import *
from scipy.sparse import issparse
from shapely import Point, Polygon
from shapely.affinity import scale as scale_func
from tqdm import tqdm

import insitupy._core.config as config
from insitupy import WITH_NAPARI, __version__
from insitupy._constants import ISPY_METADATA_FILE, LOAD_FUNCS, REGIONS_SYMBOL
from insitupy._core._checks import _check_assignment, _substitution_func
from insitupy._core._save import _save_images
from insitupy._core._xenium import (_read_binned_expression,
                                    _read_boundaries_from_xenium,
                                    _read_matrix_from_xenium,
                                    _restructure_transcripts_dataframe)
from insitupy._exceptions import UnknownOptionError
from insitupy._warnings import NoProjectLoadWarning
from insitupy.images import ImageRegistration, deconvolve_he, resize_image
from insitupy.images.utils import _get_contrast_limits
from insitupy.io.files import read_json, write_dict_to_json
from insitupy.io.io import (read_baysor_cells, read_baysor_transcripts,
                            read_celldata, read_shapesdata)
from insitupy.io.plots import save_and_show_figure
from insitupy.plotting import volcano_plot
from insitupy.utils.deg import create_deg_dataframe
from insitupy.utils.preprocessing import (normalize_and_transform_anndata,
                                          reduce_dimensions_anndata)
from insitupy.utils.utils import convert_to_list, get_nrows_maxcols

from .._constants import CACHE, ISPY_METADATA_FILE, MODALITIES
from .._exceptions import (InSituDataMissingObject,
                           InSituDataRepeatedCropError, ModalityNotFoundError,
                           NotOneElementError, WrongNapariLayerTypeError)
from ..images.utils import create_img_pyramid
from ..io.files import check_overwrite_and_remove_if_true, read_json
from ..plotting import expr_along_obs_val
from ..utils.utils import (convert_napari_shape_to_polygon_or_line,
                           convert_to_list)
from ..utils.utils import textformat as tf
from ._layers import _create_points_layer
from ._save import (_save_alt, _save_annotations, _save_cells, _save_images,
                    _save_regions, _save_transcripts)
from .dataclasses import AnnotationsData, CellData, ImageData, RegionsData

# optional packages that are not always installed
if WITH_NAPARI:
    import napari
    from napari.layers import Layer, Points, Shapes

    #from napari.layers.shapes.shapes import Shapes
    from ._layers import _add_annotations_as_layer
    from ._widgets import _initialize_widgets, add_new_geometries_widget


class InSituData:
    #TODO: Docstring of InSituData

    # import deprecated functions
    from ._deprecated import (read_all, read_annotations, read_cells,
                              read_images, read_regions, read_transcripts)

    def __init__(self,
                 path: Union[str, os.PathLike, Path] = None,
                 metadata: dict = None,
                 slide_id: str = None,
                 sample_id: str = None,
                 from_insitudata: bool = None,
                 ):
        """
        """
        # metadata
        self._path = Path(path)
        self._metadata = metadata
        self._slide_id = slide_id
        self._sample_id = sample_id
        self._from_insitudata = from_insitudata

        # modalities
        self._images = None
        self._cells = None
        self._alt = None
        self._annotations = None
        self._transcripts = None
        self._regions = None

        # other
        self._viewer = None
        self._quicksave_dir = None

    def __repr__(self):
        if self._metadata is None:
            method = "unknown"
        else:
            try:
                method = self._metadata["method"]
            except KeyError:
                method = "unknown"

        if self._path is not None:
            self._path = self._path.resolve()

        # check if all modalities are empty
        is_empty = np.all([elem is None for elem in [self._images, self._cells, self._alt, self._annotations, self._transcripts, self._regions]])

        # if is_empty:
        #     repr = f"{tf.Bold+tf.Red}InSituData{tf.ResetAll}\nEmpty"
        # else:
        repr = (
            f"{tf.Bold+tf.Red}InSituData{tf.ResetAll}\n"
            f"{tf.Bold}Method:{tf.ResetAll}\t\t{method}\n"
            f"{tf.Bold}Slide ID:{tf.ResetAll}\t{self._slide_id}\n"
            f"{tf.Bold}Sample ID:{tf.ResetAll}\t{self._sample_id}\n"
            f"{tf.Bold}Path:{tf.ResetAll}\t\t{self._path}\n"
        )

        if self._metadata is not None:
            if "metadata_file" in self._metadata:
                mfile = self._metadata["metadata_file"]
            else:
                mfile = None
        else:
            mfile = None

        repr += f"{tf.Bold}Metadata file:{tf.ResetAll}\t{mfile}"

        if is_empty:
            repr += "\n\nNo modalities loaded."
        else:
            if self._images is not None:
                images_repr = self._images.__repr__()
                repr = (
                    repr + f"\n{tf.SPACER+tf.RARROWHEAD} " + images_repr.replace("\n", f"\n{tf.SPACER}   ")
                )

            if self._cells is not None:
                cells_repr = self._cells.__repr__()
                repr = (
                    repr + f"\n{tf.SPACER+tf.RARROWHEAD+tf.Green+tf.Bold} cells{tf.ResetAll}\n{tf.SPACER}   " + cells_repr.replace("\n", f"\n{tf.SPACER}   ")
                )

            if self._transcripts is not None:
                trans_repr = f"DataFrame with shape {self._transcripts.shape[0]} x {self._transcripts.shape[1]}"

                repr = (
                    repr + f"\n{tf.SPACER+tf.RARROWHEAD+tf.Purple+tf.Bold} transcripts{tf.ResetAll}\n{tf.SPACER}   " + trans_repr
                )

            if self._annotations is not None:
                annot_repr = self._annotations.__repr__()
                repr = (
                    repr + f"\n{tf.SPACER+tf.RARROWHEAD} " + annot_repr.replace("\n", f"\n{tf.SPACER}   ")
                )

            if self._regions is not None:
                region_repr = self._regions.__repr__()
                repr = (
                    repr + f"\n{tf.SPACER+tf.RARROWHEAD} " + region_repr.replace("\n", f"\n{tf.SPACER}   ")
                )

            if self._alt is not None:
                cells_repr = self._alt.__repr__()
                altseg_keys = self._alt.keys()
                repr = (
                    #repr + f"\n{tf.SPACER+tf.RARROWHEAD+tf.Green+tf.Bold} alt{tf.ResetAll}\n{tf.SPACER}   " + cells_repr.replace("\n", f"\n{tf.SPACER}   ")
                    repr + f"\n{tf.SPACER+tf.RARROWHEAD+tf.Green+tf.Bold} alt{tf.ResetAll}\n"
                    f"{tf.SPACER}   Alternative CellData objects with following keys: {','.join(altseg_keys)}"
                )
        return repr


    @property
    def path(self):
        """Return save path of the InSituData object.
        Returns:
            str: Save path.
        """
        return self._path

    @property
    def metadata(self):
        """Return metadata of the InSituData object.
        Returns:
            dict: Metadata.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: dict):
        self._metadata = metadata

    @property
    def slide_id(self):
        """Return slide id of the InSituData object.
        Returns:
            str: Slide id.
        """
        return self._slide_id

    @property
    def sample_id(self):
        """Return sample id of the InSituData object.
        Returns:
            str: Sample id.
        """
        return self._sample_id

    @property
    def from_insitudata(self):
        return self._from_insitudata

    @property
    def images(self):
        """Return images of the InSituData object.
        Returns:
            insitupy._core.dataclasses.ImageData: Images.
        """
        return self._images

    @images.setter
    def images(self, images: ImageData):
        self._images = images

    @images.deleter
    def images(self):
        self._images = None

    @property
    def cells(self):
        """Return cell data of the InSituData object.
        Returns:
            insitupy._core.dataclasses.CellData: Cell data.
        """
        return self._cells

    @cells.setter
    def cells(self, value: CellData):
        self._cells = value

    @cells.deleter
    def cells(self):
        self._cells = None

    @property
    def transcripts(self):
        """Return transcripts of the InSituData object.
        Returns:
            pd.DataFrame: Transcripts.
        """
        return self._transcripts

    @transcripts.setter
    def transcripts(self, value: pd.DataFrame):
        self._transcripts = value

    @transcripts.deleter
    def transcripts(self):
        self._transcripts = None

    @property
    def viewer(self):
        """Return viewer of the InSituData object.
        """
        return self._viewer

    @viewer.setter
    def viewer(self, value):
        self._viewer = value

    @viewer.deleter
    def viewer(self):
        self._viewer = None

    @property
    def annotations(self):
        """Return annotations of the InSituData object.
        Returns:
            insitupy._core.dataclasses.AnnotationsData: Annotations.
        """
        return self._annotations

    @annotations.setter
    def annotations(self, value: AnnotationsData):
        self._annotations = value

    @annotations.deleter
    def annotations(self):
        self._annotations = None

    @property
    def alt(self):
        """Return alternative cell data of the InSituData object.
        Returns:
            dict: A dictionary containing `insitupy._core.dataclasses.CellData` objects..
        """
        return self._alt

    @alt.deleter
    def alt(self):
        self._alt = None

    @property
    def regions(self):
        """Return regions of the InSituData object.
        Returns:
            insitupy._core.dataclasses.RegionsData: Regions.
        """
        return self._regions

    @regions.setter
    def regions(self, value: RegionsData):
        self._regions = value

    @regions.deleter
    def regions(self):
        self._regions = None

    def assign_geometries(self,
                          geometry_type: Literal["annotations", "regions"],
                          keys: Union[str, Literal["all"]] = "all",
                          add_masks: bool = False,
                          add_to_obs: bool = False,
                          overwrite: bool = True,
                          alt_layer: str = None
                          ):
        '''
        Function to assign geometries (annotations or regions) to the anndata object in
        InSituData.cells.matrix. Assignment information is added to the DataFrame in `.obs`.
        '''
        # assert that prerequisites are met
        try:
            geom_attr = getattr(self, geometry_type)
        except AttributeError:
            raise ModalityNotFoundError(modality=geometry_type)

        if alt_layer is None:
            if self._cells is not None:
                cell_attr = self._cells
                name = ".cells"
            else:
                raise ModalityNotFoundError("cells")
        else:
            #TODO
            if self._alt is not None:
                cell_attr = self._alt[alt_layer]
                name = f".alt[{alt_layer}]"
            else:
                raise ModalityNotFoundError(f"alt[{alt_layer}]")

        if keys == "all":
            keys = geom_attr.metadata.keys()

        # make sure annotation keys are a list
        keys = convert_to_list(keys)

        # convert coordinates into shapely Point objects
        x = cell_attr.matrix.obsm["spatial"][:, 0]
        y = cell_attr.matrix.obsm["spatial"][:, 1]
        cells = gpd.points_from_xy(x, y)

        # iterate through annotation keys
        for key in keys:
            print(f"Assigning key '{key}'...")
            # extract pandas dataframe of current key
            geom_df = geom_attr[key]

            # get unique list of annotation names
            geom_names = geom_df.name.unique()

            # initiate dataframe as dictionary
            data = {}

            # iterate through names
            for n in geom_names:
                polygons = geom_df[geom_df["name"] == n]["geometry"].tolist()
                #scales = geom_df[geom_df["name"] == n]["scale"].tolist()

                # in_poly = []
                # for poly, scale in zip(polygons, scales):
                #     # scale the polygon
                #     poly = scale_func(poly, xfact=scale[0], yfact=scale[1], origin=(0,0))

                #     # check if which of the points are inside the current annotation polygon
                #     in_poly.append(poly.contains(cells))

                in_poly = [poly.contains(cells) for poly in polygons]

                # check if points were in any of the polygons
                in_poly_res = np.array(in_poly).any(axis=0)

                # collect results
                data[n] = in_poly_res

            # convert into pandas dataframe
            data = pd.DataFrame(data)
            data.index = cell_attr.matrix.obs_names

            # transform data into one column
            column_to_add = [" & ".join(geom_names[row.values]) if np.any(row.values) else "unassigned" for _, row in data.iterrows()]

            if add_to_obs:
                # create annotation from annotation masks
                col_name = f"{geometry_type}-{key}"
                data[col_name] = column_to_add

                if col_name in cell_attr.matrix.obs:
                    if overwrite:
                        cell_attr.matrix.obs.drop(col_name, axis=1, inplace=True)
                        print(f'Existing column "{col_name}" is overwritten.', flush=True)
                        add = True
                    else:
                        warn(f'Column "{col_name}" exists already in `{name}.matrix.obs`. Assignment of key "{key}" was skipped. To force assignment, select `overwrite=True`.')
                        add = False

                if add:
                    if add_masks:
                        cell_attr.matrix.obs = pd.merge(left=cell_attr.matrix.obs, right=data, left_index=True, right_index=True)
                    else:
                        cell_attr.matrix.obs = pd.merge(left=cell_attr.matrix.obs, right=data.iloc[:, -1], left_index=True, right_index=True)

                    # save that the current key was analyzed
                    geom_attr.metadata[key]["analyzed"] = tf.TICK
            else:
                # add to obsm
                obsm_keys = cell_attr.matrix.obsm.keys()
                if geometry_type not in obsm_keys:
                    # add empty pandas dataframe with obs_names as index
                    cell_attr.matrix.obsm[geometry_type] = pd.DataFrame(index=cell_attr.matrix.obs_names)

                cell_attr.matrix.obsm[geometry_type][key] = column_to_add

                # save that the current key was analyzed
                geom_attr.metadata[key]["analyzed"] = tf.TICK

                print(f"Added results to `{name}.matrix.obsm['{geometry_type}']", flush=True)


    def assign_annotations(
        self,
        keys: Union[str, Literal["all"]] = "all",
        add_masks: bool = False,
        overwrite: bool = True
    ):
        self.assign_geometries(
            geometry_type="annotations",
            keys=keys,
            add_masks=add_masks,
            overwrite=overwrite
        )
        if self._alt is not None:
            for key in self.alt.keys():
                self.assign_geometries(
                    geometry_type="annotations",
                    keys=keys,
                    add_masks=add_masks,
                    overwrite=overwrite,
                    alt_layer=key
                )

    def assign_regions(
        self,
        keys: Union[str, Literal["all"]] = "all",
        add_masks: bool = False,
        overwrite: bool = True
    ):
        self.assign_geometries(
            geometry_type="regions",
            keys=keys,
            add_masks=add_masks,
            overwrite=overwrite
        )
        if self._alt is not None:
            for key in self.alt.keys():
                self.assign_geometries(
                    geometry_type="regions",
                    keys=keys,
                    add_masks=add_masks,
                    overwrite=overwrite,
                    alt_layer=key
                )

    def copy(self):
        '''
        Function to generate a deep copy of the InSituData object.
        '''
        from copy import deepcopy
        had_viewer = False
        if self._viewer is not None:
            had_viewer = True

            # make copy of viewer to add it later again
            viewer_copy = self._viewer.copy()
            # remove viewer because there is otherwise a error during deepcopy
            self.viewer = None

        # make copy
        self_copy = deepcopy(self)

        # add viewer again to original object if necessary
        if had_viewer:
            self._viewer = viewer_copy

        return self_copy

    def crop(self,
             region_tuple: Optional[Tuple[str, str]] = None,
             xlim: Optional[Tuple[int, int]] = None,
             ylim: Optional[Tuple[int, int]] = None,
             inplace: bool = False
             #layer_name: Optional[str] = None,
            ):
        """
        Crop the data based on the provided parameters.

        Args:
            region_tuple (Optional[Tuple[str, str]]): A tuple specifying the region to crop.
            xlim (Optional[Tuple[int, int]]): The x-axis limits for cropping.
            ylim (Optional[Tuple[int, int]]): The y-axis limits for cropping.
            inplace (bool): If True, modify the data in place. Otherwise, return a new cropped data.

        Raises:
            ValueError: If none of region_tuple, layer_name, or xlim/ylim are provided.
        """
        # if layer_name is None and region_tuple is None and (xlim is None or ylim is None):
        #     raise ValueError("At least one of shape_layer, region_tuple, or xlim/ylim must be provided.")
        if region_tuple is None and (xlim is None or ylim is None):
            raise ValueError("At least one of `region_tuple` or `xlim`/`ylim` must be provided.")

        # retrieve pixel size of data
        #pixel_size = self.metadata["method_params"]["pixel_size"]

        if region_tuple is not None:

            # extract regions dataframe
            region_key = region_tuple[0]
            region_name = region_tuple[1]
            region_df = self._regions[region_key]

            # extract geometry
            geometry = region_df[region_df["name"] == region_name]["geometry"].item()

            use_shape = True

        # elif layer_name is not None:
        #     try:
        #         # extract shape layer for cropping from napari viewer
        #         layer = self.viewer.layers[layer_name]
        #     except KeyError:
        #         raise KeyError(f"Shape layer selected for cropping ('{layer_name}') was not found in layers.")

        #     # check the type of the element
        #     if not isinstance(layer, napari.layers.Shapes):
        #         raise WrongNapariLayerTypeError(found=type(layer), wanted=napari.layers.Shapes)

        #     # make sure the layer contains only one element
        #     if len(layer.data) != 1:
        #         raise NotOneElementError(layer.data)

        #     # select the shape from list
        #     crop_window = layer.data[0].copy()
        #     # crop_window *= pixel_size
        #     shape_type = layer.shape_type[0]

        #     geometry = convert_napari_shape_to_polygon_or_line(
        #         napari_shape_data=crop_window,
        #         shape_type=shape_type
        #         )

        #     use_shape = True

        else:
            # if xlim or ylim is not none, assert that both are not None
            #if xlim is not None or ylim is not None:
            assert np.all([elem is not None for elem in [xlim, ylim]]), "If `region_tuple` is None, both `xlim` and `ylim` need to be set instead."
            use_shape = False

        # # assert that either shape_layer is given or xlim/ylim
        # assert np.any([elem is not None for elem in [shape_layer, xlim, ylim]]), "No values given for either `shape_layer` or `xlim/ylim`."

        if use_shape:
            # convert to metric unit (normally µm)
            #geometry = scale_func(geometry, xfact=pixel_size, yfact=pixel_size, origin=(0,0))

            # extract x and y limits from the geometry
            bounding_box = geometry.bounds # (minx, miny, maxx, maxy)
            xlim = (bounding_box[0], bounding_box[2])
            ylim = (bounding_box[1], bounding_box[3])

            # xlim = (crop_window[:, 1].min(), crop_window[:, 1].max())
            # ylim = (crop_window[:, 0].min(), crop_window[:, 0].max())

        # make sure there are no negative values in the limits
        xlim = tuple(np.clip(xlim, a_min=0, a_max=None))
        ylim = tuple(np.clip(ylim, a_min=0, a_max=None))

        # check if the changes are supposed to be made in place or not
        if inplace:
            _self = self
        else:
            _self = self.copy()

        try:
            # if the object was previously cropped, check if the current window is identical with the previous one
            if np.all([elem in _self.metadata["method_params"].keys() for elem in ["cropping_xlim", "cropping_ylim"]]):
                # test whether the limits are identical
                if (xlim == _self.metadata["method_params"]["cropping_xlim"]) & (ylim == _self.metadata["method_params"]["cropping_ylim"]):
                    raise InSituDataRepeatedCropError(xlim, ylim)
        except TypeError:
            pass

        if _self.cells is not None:
            _self.cells.crop(
                xlim=xlim, ylim=ylim, inplace=True
            )

        if _self.alt is not None:
            alt = _self.alt
            for k, alt_cells in alt.items():
                alt_cells.crop(
                    xlim=xlim, ylim=ylim, inplace=True
                )

        if _self.transcripts is not None:
            # infer mask for selection
            xmask = (_self.transcripts["coordinates", "x"] >= xlim[0]) & (_self.transcripts["coordinates", "x"] <= xlim[1])
            ymask = (_self.transcripts["coordinates", "y"] >= ylim[0]) & (_self.transcripts["coordinates", "y"] <= ylim[1])
            mask = xmask & ymask

            # select
            _self.transcripts = _self.transcripts.loc[mask, :].copy()

            # move origin again to 0 by subtracting the lower limits from the coordinates
            _self.transcripts["coordinates", "x"] -= xlim[0]
            _self.transcripts["coordinates", "y"] -= ylim[0]

        if self._images is not None:
            _self.images.crop(xlim=xlim, ylim=ylim)

        if self._annotations is not None:

            _self.annotations.crop(
                xlim=tuple([elem for elem in xlim]),
                ylim=tuple([elem for elem in ylim])
                # xlim=tuple([elem / pixel_size for elem in xlim]), # transform back to pixel coordinates before cropping
                # ylim=tuple([elem / pixel_size for elem in ylim])
                )

        if self._regions is not None:
            _self.regions.crop(
                xlim=tuple([elem for elem in xlim]),
                ylim=tuple([elem for elem in ylim])
                # xlim=tuple([elem / pixel_size for elem in xlim]), # transform back to pixel coordinates before cropping
                # ylim=tuple([elem / pixel_size for elem in ylim])
            )

        if _self.metadata is not None:
            # add information about cropping to metadata
            if "cropping_history" not in _self.metadata:
                _self.metadata["cropping_history"] = {}
                _self.metadata["cropping_history"]["xlim"] = []
                _self.metadata["cropping_history"]["ylim"] = []
            _self.metadata["cropping_history"]["xlim"].append(tuple([int(elem) for elem in xlim]))
            _self.metadata["cropping_history"]["ylim"].append(tuple([int(elem) for elem in ylim]))

            # add new uid to uid history
            _self.metadata["uids"].append(str(uuid4()))

            # empty current data and data history entry in metadata
            _self.metadata["data"] = {}
            for k in _self.metadata["history"].keys():
                if k != "alt":
                    _self.metadata["history"][k] = []
                else:
                    empty_alt_hist_dict = {k: [] for k in _self.metadata["history"]["alt"].keys()}
                    _self.metadata["history"]["alt"] = empty_alt_hist_dict

        # sometimes modalities like annotations or regions can be empty in the meantime
        # here such empty modalities are removed
        #_self.remove_empty_modalities()

        if inplace:
            if self._viewer is not None:
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

        sc.pp.highly_variable_genes(self._cells.matrix, batch_key=hvg_batch_key, flavor=hvg_flavor, layer=hvg_layer, n_top_genes=hvg_n_top_genes)


    def normalize_and_transform(self,
                transformation_method: Literal["log1p", "sqrt"] = "log1p",
                target_sum: int = 250,
                normalize_alt: bool = True,
                verbose: bool = True
                ) -> None:
        """
        Normalize the data using either log1p or square root transformation.

        Args:
            transformation_method (Literal["log1p", "sqrt"], optional):
                The method used for data transformation. Choose between "log1p" for logarithmic transformation
                and "sqrt" for square root transformation. Default is "log1p".
            normalize_alt (bool, optional):
                If True, `.alt` modalities are also normalized, if available.
            verbose (bool, optional):
                If True, print progress messages during normalization. Default is True.

        Raises:
            ValueError: If `transformation_method` is not one of ["log1p", "sqrt"].

        Returns:
            None: This method modifies the input matrix in place, normalizing the data based on the specified method.
                It does not return any value.
        """
        if self._cells is not None:
            cells = self._cells
        else:
            raise ModalityNotFoundError(modality="cells")

        normalize_and_transform_anndata(
            adata=cells.matrix,
            transformation_method=transformation_method,
            target_sum=target_sum,
            verbose=verbose)

        if self._alt is not None:
            alt = self._alt
            print("Found `.alt` modality.")
            for k, cells in alt.items():
                print(f"\tNormalizing {k}...")
                normalize_and_transform_anndata(adata=cells.matrix, transformation_method=transformation_method, verbose=verbose)

    def add_alt(self,
                celldata_to_add: CellData,
                key_to_add: str
                ) -> None:
        # check if the current self has already an alt object and add a empty one if not
        #alt_attr_name = "alt"
        #try:
        #    alt_attr = getattr(self, alt_attr_name)
        #except AttributeError:
        #    setattr(self, alt_attr_name, {})
        #    alt_attr = getattr(self, alt_attr_name)

        if self._alt is None:
            self._alt = dict()

        # add the celldata to the given key
        self._alt[key_to_add] = celldata_to_add

    def add_baysor(self,
                   path: Union[str, os.PathLike, Path],
                   read_transcripts: bool = False,
                   key_to_add: str = "baysor",
                   pixel_size: Number = 1 # the pixel size is usually 1 since baysor runs on the µm coordinates
                   ):

        # # convert to pathlib path
        path = Path(path)

        # read baysor data
        celldata = read_baysor_cells(baysor_output=path, pixel_size=pixel_size)

        # add celldata to alt attribute
        self.add_alt(celldata_to_add=celldata, key_to_add=key_to_add)

        if read_transcripts:
            #trans_attr_name = "transcripts"
            if self._transcripts is None:
                print("No transcript layer found. Addition of Baysor transcript data is skipped.", flush=True)
            else:
                trans_attr = self._transcripts
                # read baysor transcripts
                baysor_results = read_baysor_transcripts(baysor_output=path)
                baysor_results = baysor_results[["cell"]]

                # merge transcripts with existing transcripts
                baysor_results.columns = pd.MultiIndex.from_tuples([("cell_id", key_to_add)])
                trans_attr = pd.merge(left=trans_attr,
                                    right=baysor_results,
                                    left_index=True,
                                    right_index=True
                                    )

                # add resulting dataframe to InSituData
                self._transcripts = trans_attr


    def plot_dimred(self, save: Optional[str] = None):
        '''
        Read dimensionality reduction plots.
        '''
        # construct paths
        analysis_path = self._path / "analysis"
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

    def load_all(self,
                 skip: Optional[str] = None,
                 verbose: bool = False
                 ):
        # # extract read functions
        # read_funcs = [elem for elem in dir(self) if elem.startswith("load_")]
        # read_funcs = [elem for elem in read_funcs if elem not in ["load_all", "load_quicksave"]]

        for f in LOAD_FUNCS:
            if skip is None or skip not in f:
                func = getattr(self, f)
                try:
                    func(verbose=verbose)
                except ModalityNotFoundError as err:
                    if verbose:
                        print(err)

    def load_annotations(self, verbose: bool = False):
        if verbose:
            print("Loading annotations...", flush=True)
        try:
            p = self._metadata["data"]["annotations"]
        except KeyError:
            if verbose:
                raise ModalityNotFoundError(modality="annotations")
        else:
            self._annotations = read_shapesdata(path=self._path / p, mode="annotations")


    def import_annotations(self,
                           files: Optional[Union[str, os.PathLike, Path]],
                           keys: Optional[str],
                           scale_factor: Number, # µm/pixel - can be used to convert the pixel coordinates into µm coordinates
                           verbose: bool = False
                           ):
        if verbose:
            print("Importing annotations...", flush=True)

        # add annotations object
        files = convert_to_list(files)
        keys = convert_to_list(keys)

        if self._annotations is None:
            self._annotations = AnnotationsData()

        for key, file in zip(keys, files):
            # read annotation and store in dictionary
            self._annotations.add_data(data=file,
                                      key=key,
                                      scale_factor=scale_factor
                                      )

        #self._remove_empty_modalities()

    def load_regions(self, verbose: bool = False):
        if verbose:
            print("Loading regions...", flush=True)
        try:
            p = self._metadata["data"]["regions"]
        except KeyError:
            if verbose:
                raise ModalityNotFoundError(modality="regions")
        else:
            self._regions = read_shapesdata(path=self._path / p, mode="regions")

    def import_regions(self,
                    files: Optional[Union[str, os.PathLike, Path]],
                    keys: Optional[str],
                    scale_factor: Number, # µm/pixel - used to convert the pixel coordinates into µm coordinates
                    verbose: bool = False
                    ):
        if verbose:
            print("Importing regions...", flush=True)

        # add regions object
        files = convert_to_list(files)
        keys = convert_to_list(keys)
        #pixel_size = self.metadata["method_params"]['pixel_size']

        if self._regions is None:
            self._regions = RegionsData()

        for key, file in zip(keys, files):
            # read annotation and store in dictionary
            self._regions.add_data(data=file,
                                key=key,
                                scale_factor=scale_factor
                                )

        #self._remove_empty_modalities()


    def load_cells(self, verbose: bool = False):
        if verbose:
            print("Loading cells...", flush=True)

        if self._from_insitudata:
            try:
                cells_path = self._metadata["data"]["cells"]
            except KeyError:
                if verbose:
                    raise ModalityNotFoundError(modality="cells")
            else:
                self._cells = read_celldata(path=self._path / cells_path)

            # check if alt data is there and read if yes
            try:
                alt_path_dict = self._metadata["data"]["alt"]
            except KeyError:
                if verbose:
                    print("\tNo alternative cells found...")
            else:
                print("\tFound alternative cells...")
                alt_dict = {}
                for k, p in alt_path_dict.items():
                    alt_dict[k] = read_celldata(path=self._path / p)

                # add attribute
                self._alt = alt_dict
        else:
            NoProjectLoadWarning()

    def load_images(self,
                    names: Union[Literal["all", "nuclei"], str] = "all", # here a specific image can be chosen
                    overwrite: bool = False,
                    verbose: bool = False
                    ):
        # load image into ImageData object
        if verbose:
            print("Loading images...", flush=True)

        if self._from_insitudata:
            # check if matrix data is stored in this InSituData
            try:
                images_dict = self._metadata["data"]["images"]
            except KeyError:
                if verbose:
                    raise ModalityNotFoundError(modality="images")
            else:
                if names == "all":
                    img_names = list(images_dict.keys())
                else:
                    img_names = convert_to_list(names)

                # get file paths and names
                img_files = [v for k,v in images_dict.items() if k in img_names]
                img_names = [k for k,v in images_dict.items() if k in img_names]

                # create imageData object
                img_paths = [self._path / elem for elem in img_files]

                if self._images is None:
                    self._images = ImageData(img_paths, img_names)
                else:
                    for im, n in zip(img_paths, img_names):
                        self._images.add_image(im, n, overwrite=overwrite, verbose=verbose)

        else:
            NoProjectLoadWarning()

    def load_transcripts(self,
                        verbose: bool = False
                        ):
        # read transcripts
        if verbose:
            print("Loading transcripts...", flush=True)

        if self._from_insitudata:
            # check if matrix data is stored in this InSituData
            try:
                transcripts_path = self._metadata["data"]["transcripts"]
            except KeyError:
                if verbose:
                    raise ModalityNotFoundError(modality="transcripts")
            else:
                self._transcripts = pd.read_parquet(self._path / transcripts_path)

        else:
            NoProjectLoadWarning()

    @classmethod
    def read(cls, path: Union[str, os.PathLike, Path]):
        """Read an InSituData object from a specified folder.

        Args:
            path (Union[str, os.PathLike, Path]): The path to the folder where data is saved.

        Returns:
            InSituData: A new InSituData object with the loaded data.
        """
        path = Path(path) # make sure the path is a pathlib path
        assert (path / ISPY_METADATA_FILE).exists(), "No insitupy metadata file found."
        # read InSituData metadata
        insitupy_metadata_file = path / ISPY_METADATA_FILE
        metadata = read_json(insitupy_metadata_file)

        # retrieve slide_id and sample_id
        slide_id = metadata["slide_id"]
        sample_id = metadata["sample_id"]

        # save paths of this project in metadata
        metadata["path"] = abspath(path).replace("\\", "/")
        metadata["metadata_file"] = ISPY_METADATA_FILE

        data = cls(path=path,
                   metadata=metadata,
                   slide_id=slide_id,
                   sample_id=sample_id,
                   from_insitudata=True
                   )
        return data

    def reduce_dimensions(self,
                        umap: bool = True,
                        tsne: bool = True,
                        layer: Optional[str] = None,
                        batch_correction_key: Optional[str] = None,
                        perform_clustering: bool = True,
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
            layer (str, optional):
                Specifies the layer of the AnnData object to operate on. Default is None (uses adata.X).
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
        if self._cells is None:
            raise ModalityNotFoundError(modality="cells")
        else:
            cells = self._cells

        reduce_dimensions_anndata(adata=cells.matrix,
                                  umap=umap, tsne=tsne, layer=layer,
                                  batch_correction_key=batch_correction_key,
                                  perform_clustering=perform_clustering,
                                  verbose=verbose,
                                  tsne_lr=tsne_lr, tsne_jobs=tsne_jobs
                                  )

        if self._alt is not None:
            alt = self._alt
            print("Found `.alt` modality.")
            for k, cells in alt.items():
                print(f"\tReducing dimensions in `.alt['{k}']...")

                reduce_dimensions_anndata(adata=cells.matrix,
                                        umap=umap, tsne=tsne, layer=layer,
                                        batch_correction_key=batch_correction_key,
                                        perform_clustering=perform_clustering,
                                        verbose=verbose,
                                        tsne_lr=tsne_lr, tsne_jobs=tsne_jobs
                                        )

    def saveas(self,
            path: Union[str, os.PathLike, Path],
            overwrite: bool = False,
            zip_output: bool = False,
            images_as_zarr: bool = True,
            zarr_zipped: bool = False,
            images_max_resolution: Optional[Number] = None, # in µm per pixel
            verbose: bool = True
            ):
        '''
        Function to save the InSituData object.

        Args:
            path: Path to save the data to.
        '''
        # check if the path already exists
        path = Path(path)

        # check overwrite
        check_overwrite_and_remove_if_true(path=path, overwrite=overwrite)

        if zip_output:
            zippath = path / (path.stem + ".zip")
            check_overwrite_and_remove_if_true(path=zippath, overwrite=overwrite)

        print(f"Saving data to {str(path)}") if verbose else None

        # create output directory if it does not exist yet
        path.mkdir(parents=True, exist_ok=True)

        # store basic information about experiment
        self._metadata["slide_id"] = self._slide_id
        self._metadata["sample_id"] = self._sample_id

        # clean old entries in data metadata
        self._metadata["data"] = {}

        # save images
        if self._images is not None:
            images = self._images
            _save_images(
                imagedata=images,
                path=path,
                metadata=self._metadata,
                images_as_zarr=images_as_zarr,
                zipped=zarr_zipped,
                max_resolution=images_max_resolution,
                verbose=False
                )

        # save cells
        if self._cells is not None:
            cells = self._cells
            _save_cells(
                cells=cells,
                path=path,
                metadata=self._metadata,
                boundaries_zipped=zarr_zipped
            )

        # save alternative cell data
        if self._alt is not None:
            alt = self._alt
            _save_alt(
                attr=alt,
                path=path,
                metadata=self._metadata,
                boundaries_zipped=zarr_zipped
            )

        # save transcripts
        if self._transcripts is not None:
            transcripts = self._transcripts
            _save_transcripts(
                transcripts=transcripts,
                path=path,
                metadata=self._metadata
                )

        # save annotations
        if self._annotations is not None:
            annotations = self._annotations
            _save_annotations(
                annotations=annotations,
                path=path,
                metadata=self._metadata
            )

        # save regions
        if self._regions is not None:
            regions = self._regions
            _save_regions(
                regions=regions,
                path=path,
                metadata=self._metadata
            )

        # save version of InSituPy
        self._metadata["version"] = __version__

        if "method_params" in self._metadata:
            # move method_param key to end of metadata
            self._metadata["method_params"] = self._metadata.pop("method_params")

        # write Xeniumdata metadata to json file
        xd_metadata_path = path / ISPY_METADATA_FILE
        write_dict_to_json(dictionary=self._metadata, file=xd_metadata_path)

        # Optionally: zip the resulting directory
        if zip_output:
            shutil.make_archive(path, 'zip', path, verbose=False)
            shutil.rmtree(path) # delete directory

        # change path to the new one
        self._path = path.resolve()

        # reload the modalities
        self.reload(verbose=False)

        print("Saved.") if verbose else None

    def save(self,
             path: Optional[Union[str, os.PathLike, Path]] = None,
             zarr_zipped: bool = False,
             verbose: bool = True
             ):

        # check path
        if path is not None:
            path = Path(path)
        else:
            if self._from_insitudata:
                path = Path(self._metadata["path"])
            else:
                warn(
                    f"Data as not loaded from an InSituPy project. "
                    f"Use `saveas()` instead to save the data to a new project folder."
                    )

        if path.exists():
            # check if path is a valid directory
            if not path.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {str(path)}")

            # check if the folder is a InSituPy project
            metadata_file = path / ISPY_METADATA_FILE

            if metadata_file.exists():
                # read metadata file and check uid
                project_meta = read_json(metadata_file)

                # check uid
                project_uid = project_meta["uids"][-1]  # [-1] to select latest uid
                current_uid = self._metadata["uids"][-1]
                if current_uid == project_uid:
                    self._update_to_existing_project(path=path,
                                                     zarr_zipped=zarr_zipped,
                                                     verbose=verbose
                                                     )

                    # reload the modalities
                    self.reload(verbose=False, skip=["transcripts", "images"])
                else:
                    warn(
                        f"UID of current object {current_uid} not identical with UID in project path {path}: {project_uid}.\n"
                        f"Project is neither saved nor updated. Try `saveas()` instead to save the data to a new project folder. "
                        f"A reason for this could be the data has been cropped in the meantime."
                    )
            else:
                warn(
                    f"No `.ispy` metadata file in {path}. Directory is probably no valid InSituPy project. "
                    f"Use `saveas()` instead to save the data to a new InSituPy project."
                    )


        else:
            # save to the respective directory
            self.saveas(path=path)

    def save_current_colorlegend(self, savepath):

        # Check if static_canvas exists
        if not hasattr(config, 'static_canvas'):
            print("Warning: 'static_canvas' attribute not found in config. "
                "Please display data in the napari viewer using '.show()' first.")
            return

        try:
            # Save the figure to a PDF file
            config.static_canvas.figure.savefig(savepath)
            print(f"Figure saved as {savepath}")
        except RuntimeError as e:
            if 'FigureCanvasQTAgg has been deleted' in str(e):
                print("Warning: The color legend has been deleted and cannot be saved.")
            else:
                raise  # Re-raise the exception if it's a different error

    def _update_to_existing_project(self,
                                    path: Optional[Union[str, os.PathLike, Path]],
                                    zarr_zipped: bool = False,
                                    verbose: bool = True
                                    ):
        if verbose:
            print(f"Updating project in {path}")

        # save cells
        if self._cells is not None:
            cells = self._cells
            if verbose:
                print("\tUpdating cells...", flush=True)
            _save_cells(
                cells=cells,
                path=path,
                metadata=self._metadata,
                boundaries_zipped=zarr_zipped,
                overwrite=True
            )

        # save alternative cell data
        if self._alt is not None:
            alt = self._alt
            if verbose:
                print("\tUpdating alternative segmentations...", flush=True)
            _save_alt(
                attr=alt,
                path=path,
                metadata=self._metadata,
                boundaries_zipped=zarr_zipped
            )

        # save annotations
        if self._annotations is not None:
            annotations = self._annotations
            if verbose:
                print("\tUpdating annotations...", flush=True)
            _save_annotations(
                annotations=annotations,
                path=path,
                metadata=self._metadata
            )

        # save regions
        if self._regions is not None:
            regions = self._regions
            if verbose:
                print("\tUpdating regions...", flush=True)
            _save_regions(
                regions=regions,
                path=path,
                metadata=self._metadata
            )

        # save version of InSituPy
        self._metadata["version"] = __version__

        if "method_params" in self._metadata:
            # move method_params key to end of metadata
            self._metadata["method_params"] = self._metadata.pop("method_params")

        # write Xeniumdata metadata to json file
        xd_metadata_path = path / ISPY_METADATA_FILE
        write_dict_to_json(dictionary=self._metadata, file=xd_metadata_path)

        if verbose:
            print("Saved.")


    def quicksave(self,
                  note: Optional[str] = None
                  ):
        # create quicksave directory if it does not exist already
        self._quicksave_dir = CACHE / "quicksaves"
        self._quicksave_dir.mkdir(parents=True, exist_ok=True)

        # save annotations
        if self._annotations is None:
            print("No annotations found. Quicksave skipped.", flush=True)
        else:
            annotations = self._annotations
            # create filename
            current_datetime = datetime.now().strftime("%y%m%d_%H-%M-%S")
            slide_id = self._slide_id
            sample_id = self._sample_id
            uid = str(uuid4())[:8]

            # create output directory
            outname = f"{slide_id}__{sample_id}__{current_datetime}__{uid}"
            outdir = self._quicksave_dir / outname

            _save_annotations(
                annotations=annotations,
                path=outdir,
                metadata=None
            )

            if note is not None:
                with open(outdir / "note.txt", "w") as notefile:
                    notefile.write(note)

            # # # zip the output
            # shutil.make_archive(outdir, format='zip', root_dir=outdir, verbose=False)
            # shutil.rmtree(outdir) # delete directory


    def list_quicksaves(self):
        pattern = "{slide_id}__{sample_id}__{savetime}__{uid}"

        # collect results
        res = {
            "slide_id": [],
            "sample_id": [],
            "savetime": [],
            "uid": [],
            "note": []
        }
        for d in self._quicksave_dir.glob("*"):
            parse_res = parse(pattern, d.stem).named
            for key, value in parse_res.items():
                res[key].append(value)

            notepath = d / "note.txt"
            if notepath.exists():
                with open(notepath, "r") as notefile:
                    res["note"].append(notefile.read())
            else:
                res["note"].append("")

        # create and return dataframe
        return pd.DataFrame(res)

    def load_quicksave(self,
                       uid: str
                       ):
        # find files with the uid
        files = list(self._quicksave_dir.glob(f"*{uid}*"))

        if len(files) == 1:
            ad = read_shapesdata(files[0] / "annotations", mode="annotations")
        elif len(files) == 0:
            print(f"No quicksave with uid '{uid}' found. Use `.list_quicksaves()` to list all available quicksaves.")
        else:
            raise ValueError(f"More than one quicksave with uid '{uid}' found.")

        # add annotations to existing annotations attribute or add a new one
        if self._annotations is None:
            self._annotations = AnnotationsData()
        else:
            annotations = self._annotations
            for k in ad.metadata.keys():
                annotations.add_data(ad[k], k, verbose=True)


    def show(self,
        keys: Optional[str] = None,
        # annotation_keys: Optional[str] = None,
        point_size: int = 8,
        scalebar: bool = True,
        #pixel_size: float = None, # if none, extract from metadata
        unit: str = "µm",
        # cmap_annotations: str ="Dark2",
        grayscale_colormap: List[str] = ["red", "green", "cyan", "magenta", "yellow", "gray"],
        return_viewer: bool = False,
        widgets_max_width: int = 500
        ):

        # create viewer
        self._viewer = napari.Viewer(title=f"{self._slide_id}: {self._sample_id}")

        if self._images is None:
            warn("No attribute `.images` found.")
        else:
            images_attr = self._images
            n_images = len(images_attr.metadata)
            n_grayscales = 0 # number of grayscale images
            for i, (img_name, img_metadata) in enumerate(images_attr.metadata.items()):
            #for i, img_name in enumerate(image_keys):
                img = images_attr[img_name]
                is_visible = False if i < n_images - 1 else True # only last image is set visible
                pixel_size = img_metadata['pixel_size']

                # check if the current image is RGB
                is_rgb = self._images.metadata[img_name]["rgb"]

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


                if not isinstance(img, list):
                    # create image pyramid for lazy loading
                    img_pyramid = create_img_pyramid(img=img, nsubres=6)
                else:
                    img_pyramid = img

                # infer contrast limits
                contrast_limits = _get_contrast_limits(img_pyramid)

                if contrast_limits[1] == 0:
                    warn("The maximum value of the image is 0. Is the image really completely empty?")
                    contrast_limits = (0, 255)

                # add img pyramid to napari viewer
                self._viewer.add_image(
                        img_pyramid,
                        name=img_name,
                        colormap=cmap,
                        blending=blending,
                        rgb=is_rgb,
                        contrast_limits=contrast_limits,
                        scale=(pixel_size, pixel_size),
                        visible=is_visible
                    )

        # optionally: add cells as points
        #if show_cells or keys is not None:
        if keys is not None:
            if self._cells is None:
                raise InSituDataMissingObject("cells")
            else:
                cells = self._cells
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
                    #pvis = False if i < len(keys) - 1 else True # only last image is set visible
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
                        visible=True
                    )

                    # add layer programmatically - does not work for all types of layers
                    # see: https://forum.image.sc/t/add-layerdatatuple-to-napari-viewer-programmatically/69878
                    self._viewer.add_layer(Layer.create(*layer))

        # WIDGETS
        if self._cells is None:
            # add annotation widget to napari
            add_geom_widget = add_new_geometries_widget()
            add_geom_widget.max_height = 120
            add_geom_widget.max_width = widgets_max_width
            self._viewer.window.add_dock_widget(add_geom_widget, name="Add geometries", area="right")
        else:
            cells = self._cells
            # initialize the widgets
            show_points_widget, locate_cells_widget, show_geometries_widget, show_boundaries_widget, select_data, filter_cells_widget = _initialize_widgets(xdata=self)

            # add widgets to napari window
            if select_data is not None:
                self._viewer.window.add_dock_widget(select_data, name="Select data", area="right")
                select_data.max_height = 50
                select_data.max_width = widgets_max_width

            if show_points_widget is not None:
                self.viewer.window.add_dock_widget(show_points_widget, name="Show data", area="right")
                show_points_widget.max_height = 170
                show_points_widget.max_width = widgets_max_width



            if show_boundaries_widget is not None:
                self._viewer.window.add_dock_widget(show_boundaries_widget, name="Show boundaries", area="right")
                #show_boundaries_widget.max_height = 80
                show_boundaries_widget.max_width = widgets_max_width

            if locate_cells_widget is not None:
                self._viewer.window.add_dock_widget(locate_cells_widget, name="Navigate to cell", area="right", tabify=True)
                #locate_cells_widget.max_height = 130
                locate_cells_widget.max_width = widgets_max_width

            if filter_cells_widget is not None:
                self.viewer.window.add_dock_widget(filter_cells_widget, name="Filter cells", area="right", tabify=True)
                filter_cells_widget.max_height = 150
                show_points_widget.max_width = widgets_max_width

            # add annotation widget to napari
            add_geom_widget = add_new_geometries_widget()
            #annot_widget.max_height = 100
            add_geom_widget.max_width = widgets_max_width
            self._viewer.window.add_dock_widget(add_geom_widget, name="Add geometries", area="right", add_vertical_stretch=True)

            # if show_region_widget is not None:
            #     self.viewer.window.add_dock_widget(show_region_widget, name="Show regions", area="right")
            #     show_region_widget.max_height = 100
            #     show_region_widget.max_width = widgets_max_width

            if show_geometries_widget is not None:
                self._viewer.window.add_dock_widget(show_geometries_widget, name="Show geometries", area="right", tabify=True)
                show_geometries_widget.max_width = widgets_max_width

        # EVENTS
        # Assign function to an layer addition event
        def _update_uid(event):
            if event is not None:

                layer = event.source
                if event.action == "add":
                    if 'uid' in layer.properties:
                        layer.properties['uid'][-1] = str(uuid4())
                    else:
                        layer.properties['uid'] = np.array([str(uuid4())], dtype='object')

                elif event.action == "remove":
                    pass
                else:
                    raise ValueError("Unexpected value '{event.action}' for `event.action`. Expected 'add' or 'remove'.")

        # Assign the function to data of all existing layers
        for layer in self._viewer.layers:
            if isinstance(layer, Shapes) or isinstance(layer, Points):
                layer.events.data.connect(_update_uid)

        # Connect the function to the data of existing shapes and points layers in the viewer
        def connect_to_all_shapes_layers(event):
            layer = event.source[event.index]
            if event is not None:
                if isinstance(layer, Shapes) or isinstance(layer, Points):
                    layer.events.data.connect(_update_uid)

        # Connect the function to any new layers added to the viewer
        self._viewer.layers.events.inserted.connect(connect_to_all_shapes_layers)

        # add color legend widget
        import insitupy._core.config as config
        from insitupy._core.config import init_colorlegend_canvas
        init_colorlegend_canvas()
        self._viewer.window.add_dock_widget(config.static_canvas, area='left', name='Color legend')

        # def update_colorlegend(event):
        #     # if event.type == "inserted":
        #     layer = event.source[event.index]
        #     _add_colorlegend_to_canvas(layer=layer, static_canvas=config.static_canvas)
        #     # if event.type == "removed":
        #         # config.static_canvas.figure.clear()
        #         # config.static_canvas.draw()

        # self.viewer.layers.events.inserted.connect(update_colorlegend)
        # #self.viewer.layers.events.removed.connect(add_colorlegend_widget)

        # NAPARI SETTINGS
        if scalebar:
            # add scale bar
            self._viewer.scale_bar.visible = True
            self._viewer.scale_bar.unit = unit

        napari.run()
        if return_viewer:
            return self._viewer

    def store_geometries(self,
                         name_pattern = "{type_symbol} {class_name} ({annot_key})",
                         uid_col: str = "id"
                         ):
        """
        Extracts geometric layers from shapes and points layers in the napari viewer
        and stores them in the InSituData object as annotations or regions.

        Args:
            name_pattern (str): A format string used to parse the layer names.
                It should contain placeholders for 'type_symbol', 'class_name',
                and 'annot_key'.
            uid_col (str): The name of the column used to store unique identifiers
                for the geometries. Default is "id".

        Raises:
            AttributeError: If the viewer is not initialized, an error message
                prompts the user to open a napari viewer using the `.show()` method.

        Notes:
            - The function iterates through the layers in the viewer and checks if
            they are instances of Shapes or Points.
            - It extracts the geometric data, colors, and other relevant properties
            to create a GeoDataFrame.
            - The GeoDataFrame is then added to the annotations or regions of the
            InSituData object based on the type of layer.
            - If the layer is classified as a region but is a point layer, a warning
            is issued, and the layer is skipped.
        """
        if self._viewer is not None:
            viewer = self._viewer
        else:
            print("Use `.show()` first to open a napari viewer.")

        # iterate through layers and save them as annotation or region if they meet requirements
        layers = viewer.layers
        #collection_dict = {}
        for layer in layers:
            if isinstance(layer, Shapes) or isinstance(layer, Points):
                name_parsed = parse(name_pattern, layer.name)
                if name_parsed is not None:
                    type_symbol = name_parsed.named["type_symbol"]
                    annot_key = name_parsed.named["annot_key"]
                    class_name = name_parsed.named["class_name"]

                    # if the InSituData object does not have an annotations attribute, initialize it
                    if self._annotations is None:
                        self._annotations = AnnotationsData() # initialize empty object

                    # extract shapes coordinates and colors
                    layer_data = layer.data
                    colors = layer.edge_color.tolist()
                    scale = layer.scale

                    checks_passed = True
                    is_region_layer = False
                    object_type = "annotation"
                    if type_symbol == REGIONS_SYMBOL:
                        is_region_layer = True
                        object_type = "region"
                        if isinstance(layer, Points):
                            warn(f'Layer "{layer.name}" is a point layer and at the same time classified as "Region". This is not allowed. Skipped this layer.')
                            checks_passed = False

                    if object_type == "annotation":
                        # if the InSituData object does not have an annotations attribute, initialize it
                        if not hasattr(self, "annotations"):
                            self.annotations = AnnotationsData() # initialize empty object
                    else:
                        # if the InSituData object does not have an regions attribute, initialize it
                        if not hasattr(self, "regions"):
                            self.regions = RegionsData() # initialize empty object

                    if checks_passed:
                        if isinstance(layer, Shapes):
                            # extract shape types
                            shape_types = layer.shape_type
                            # build annotation GeoDataFrame
                            geom_df = {
                                uid_col: layer.properties["uid"],
                                "objectType": object_type,
                                #"geometry": [Polygon(np.stack([ar[:, 1], ar[:, 0]], axis=1)) for ar in layer_data],  # switch x/y
                                "geometry": [convert_napari_shape_to_polygon_or_line(napari_shape_data=ar, shape_type=st) for ar, st in zip(layer_data, shape_types)],
                                "name": class_name,
                                "color": [[int(elem[e]*255) for e in range(3)] for elem in colors],
                                #"scale": [scale] * len(layer_data),
                                #"layer_type": ["Shapes"] * len(layer_data)
                            }

                        elif isinstance(layer, Points):
                            # build annotation GeoDataFrame
                            geom_df = {
                                uid_col: layer.properties["uid"],
                                "objectType": object_type,
                                "geometry": [Point(d[1], d[0]) for d in layer_data],  # switch x/y
                                "name": class_name,
                                "color": [[int(elem[e]*255) for e in range(3)] for elem in colors],
                                #"scale": [scale] * len(layer_data),
                                #"layer_type": ["Points"] * len(layer_data)
                            }

                        # generate GeoDataFrame
                        geom_df = GeoDataFrame(geom_df, geometry="geometry")

                        if is_region_layer:
                            if self._regions is None:
                                self._regions = RegionsData()

                            # add regions
                            self._regions.add_data(data=geom_df,
                                                  key=annot_key,
                                                  verbose=True,
                                                  scale_factor=scale[0]
                                                  )
                        else:
                            if self._annotations is None:
                                self._annotations = AnnotationsData()

                            # add annotations
                            self._annotations.add_data(data=geom_df,
                                                      key=annot_key,
                                                      verbose=True,
                                                      scale_factor=scale[0]
                                                      )

            else:
                pass

        #self._remove_empty_modalities()

    def plot_binned_expression(
        self,
        genes: Union[List[str], str],
        maxcols: int = 4,
        figsize: Tuple[int, int] = (8,6),
        savepath: Union[str, os.PathLike, Path] = None,
        save_only: bool = False,
        dpi_save: int = 300,
        show: bool = True,
        fontsize: int = 28
        ):
        # extract binned expression matrix and gene names
        binex = self._cells.matrix.varm["binned_expression"]
        gene_names = self._cells.matrix.var_names

        genes = convert_to_list(genes)

        nplots, nrows, ncols = get_nrows_maxcols(len(genes), max_cols=maxcols)

        # setup figure
        fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))

        # scale font sizes
        plt.rcParams.update({'font.size': fontsize})

        if nplots > 1:
            axs = axs.ravel()
        else:
            axs = [axs]

        for i, gene in enumerate(genes):
            # retrieve binned expression
            img = binex[gene_names.get_loc(gene)]

            # determine upper limit for color
            vmax = np.percentile(img[img>0], 95)

            # plot expression
            axs[i].imshow(img, cmap="viridis", vmax=vmax)

            # set title
            axs[i].set_title(gene)

        if nplots > 1:

            # check if there are empty plots remaining
            while i < nrows * maxcols - 1:
                i+=1
                # remove empty plots
                axs[i].set_axis_off()

        if show:
            fig.tight_layout()
            save_and_show_figure(savepath=savepath, fig=fig, save_only=save_only, dpi_save=dpi_save)
        else:
            return fig, axs

    def plot_expr_along_obs_val(
        self,
        keys: str,
        obs_val: str,
        groupby: Optional[str] = None,
        method: Literal["lowess", "loess"] = 'loess',
        stderr: bool = False,
        savepath=None,
        return_data=False,
        **kwargs
        ):
        # retrieve anndata object from InSituData
        adata = self._cells.matrix

        results = expr_along_obs_val(
            adata=adata,
            keys=keys,
            obs_val=obs_val,
            groupby=groupby,
            method=method,
            stderr=stderr,
            savepath=savepath,
            return_data=return_data
            **kwargs
            )

        if return_data:
            return results

    def reload(
        self,
        skip: Optional[List] = None,
        verbose: bool = True
        ):
        data_meta = self._metadata["data"]
        current_modalities = [m for m in MODALITIES if getattr(self, m) is not None and m in data_meta]

        if skip is not None:
            # remove the modalities which are supposed to be skipped during reload
            skip = convert_to_list(skip)
            for s in skip:
                try:
                    current_modalities.remove(s)
                except ValueError:
                    pass

        if len(current_modalities) > 0:
            print(f"Reloading following modalities: {', '.join(current_modalities)}") if verbose else None
            for cm in current_modalities:
                func = getattr(self, f"load_{cm}")
                func(verbose=verbose)
        else:
            print("No modalities with existing save path found. Consider saving the data with `saveas()` first.")

    def remove_history(self,
                       verbose: bool = True
                       ):

        for cat in ["annotations", "cells", "regions"]:
            dirs_to_remove = []
            #if hasattr(self, cat):
            files = sorted((self._path / cat).glob("*"))
            if len(files) > 1:
                dirs_to_remove = files[:-1]

                for d in dirs_to_remove:
                    shutil.rmtree(d)

                print(f"Removed {len(dirs_to_remove)} entries from '.{cat}'.") if verbose else None
            else:
                print(f"No history found for '{cat}'.") if verbose else None

    def remove_modality(self,
                        modality: str
                        ):
        if hasattr(self, modality):
            # delete attribute from InSituData object
            delattr(self, modality)

            # delete metadata
            self.metadata["data"].pop(modality, None) # returns None if key does not exist

        else:
            print(f"No modality '{modality}' found. Nothing removed.")


def calc_distance_of_cells_from(
    data: InSituData,
    annotation_key: str,
    annotation_class: str,
    region_key: Optional[str] = None,
    region_name: Optional[str] = None,
    key_to_save: Optional[str] = None
    ):

    """
    Calculate the distance of cells from a specified annotation class within a given region and save the results.

    This function calculates the distance of each cell in the spatial data to the closest point
    of a specified annotation class. The distances are then saved in the cell data matrix.

    Args:
        data (InSituData): The input data containing cell and annotation information.
        annotation_key (str): The key to retrieve the annotation information.
        annotation_class (Optional[str]): The specific annotation class to calculate distances from.
        region_key: (Optional[str]): If not None, `region_key` is used together with `region_name` to determine the region in which cells are considered
                                     for the analysis.
        region_name: (Optional[str]): If not None, `region_name` is used together with `region_key` to determine the region in which cells are considered
                                     for the analysis.
        key_to_save (Optional[str]): The key under which to save the calculated distances in the cell data matrix.
                                     If None, a default key is generated based on the annotation class.

    Returns:
        None
    """
    # extract anndata object
    adata = data.cells.matrix

    if region_name is None:
        print(f'Calculate the distance of cells from the annotation "{annotation_class}"')
        region_mask = [True] * len(adata)
    else:
        assert region_key is not None, "`region_key` must not be None if `region_name` is not None."
        print(f'Calculate the distance of cells from the annotation "{annotation_class}" within region "{region_name}"')

        try:
            region_df = adata.obsm["regions"]
        except KeyError:
            data.assign_regions(keys=region_key)
            region_df = adata.obsm["regions"]
        else:
            if region_key not in region_df.columns:
                data.assign_regions(keys=region_key)

        # generate mask for selected region
        region_mask = region_df[region_key] == region_name

    # create geopandas points from cells
    x = adata.obsm["spatial"][:, 0][region_mask]
    y = adata.obsm["spatial"][:, 1][region_mask]
    indices = adata.obs_names[region_mask]
    cells = gpd.points_from_xy(x, y)

    # retrieve annotation information
    annot_df = data.annotations[annotation_key]
    class_df = annot_df[annot_df["name"] == annotation_class]

    # calculate distance of cells to their closest point
    # scaled_geometries = [
    #     scale_func(geometry, xfact=scale[0], yfact=scale[1], origin=(0,0))
    #     for geometry, scale in zip(class_df["geometry"], class_df["scale"])
    #     ]
    scaled_geometries = class_df["geometry"].tolist()
    dists = np.array([cells.distance(geometry) for geometry in scaled_geometries])
    min_dists = dists.min(axis=0)

    # add indices to minimum distances
    min_dists = pd.Series(min_dists, index=indices)

    # add results to CellData
    if key_to_save is None:
        #key_to_save = f"dist_from_{annotation_class}"
        key_to_save = annotation_class
    #adata.obs[key_to_save] = min_dists

    obsm_keys = adata.obsm.keys()
    if "distance_from" not in obsm_keys:
        # add empty pandas dataframe with obs_names as index
        adata.obsm["distance_from"] = pd.DataFrame(index=adata.obs_names)

    adata.obsm["distance_from"][key_to_save] = min_dists
    print(f'Saved distances to `.cells.matrix.obsm["distance_from"]["{key_to_save}"]`')

def differential_gene_expression(
    data: InSituData,
    data_annotation_tuple: Optional[Tuple[str, str]] = None,
    ref_data: Optional[InSituData] = None,
    ref_annotation_tuple: Optional[Union[Literal["rest"], Tuple[str, str]]] = None,
    obs_tuple: Optional[Tuple[str, str]] = None,
    region_tuple: Optional[Tuple[str, str]] = None,
    plot_volcano: bool = True,
    method: Optional[Literal['logreg', 't-test', 'wilcoxon', 't-test_overestim_var']] = 't-test',
    ignore_duplicate_assignments: bool = False,
    force_assignment: bool = False,
    title: Optional[str] = None,
    savepath: Union[str, os.PathLike, Path] = None,
    save_only: bool = False,
    dpi_save: int = 300,
    **kwargs
):
    """
    Perform differential gene expression analysis on in situ sequencing data.

    This function compares gene expression between specified annotations within a single
    InSituData object or between two InSituData objects. It supports various statistical
    methods for differential expression analysis and can generate a volcano plot of the results.

    Args:
        data (InSituData): The primary in situ data object.
        data_annotation_tuple (Tuple[str, str]): Tuple containing the annotation key and name.
        ref_data (Optional[InSituData], optional): Reference in situ data object for comparison. Defaults to None.
        ref_annotation_tuple (Union[Literal["rest"], Tuple[str, str]], optional): Tuple containing the reference annotation key and name, or "rest" to use the rest of the data as reference. Defaults to "rest".
        obs_tuple (Optional[Tuple[str, str]], optional): Tuple specifying an observation key and value to filter the data. Defaults to None.
        region_tuple (Optional[Tuple[str, str]], optional): Tuple specifying a region key and name to restrict the analysis to a specific region. Defaults to None.
        plot_volcano (bool, optional): Whether to generate a volcano plot of the results. Defaults to True.
        method (Optional[Literal['logreg', 't-test', 'wilcoxon', 't-test_overestim_var']], optional): Statistical method to use for differential expression analysis. Defaults to 't-test'.
        ignore_duplicate_assignments (bool, optional): Whether to ignore duplicate assignments in the data. Defaults to False.
        force_assignment (bool, optional): Whether to force assignment of annotations and regions. Defaults to False.
        title (Optional[str], optional): Title for the volcano plot. Defaults to None.
        savepath (Union[str, os.PathLike, Path], optional): Path to save the plot (default is None).
        save_only (bool): If True, only save the plot without displaying it (default is False).
        dpi_save (int): Dots per inch (DPI) for saving the plot (default is 300).
        **kwargs: Additional keyword arguments for the volcano plot.

    Returns:
        Union[None, Dict[str, Any]]: If `plot_volcano` is True, returns None. Otherwise, returns a dictionary with the results DataFrame and parameters used for the analysis.

    Raises:
        ValueError: If `ref_annotation_tuple` is neither 'rest' nor a 2-tuple.
        AssertionError: If `ref_data` is provided when `ref_annotation_tuple` is 'rest'.
        AssertionError: If `region_tuple` is provided when `ref_data` is not None.
        AssertionError: If the specified region or annotation is not found in the data.

    Example:
        >>> result = differential_gene_expression(
                data=my_data,
                data_annotation_tuple=("cell_type", "neuron"),
                ref_data=my_ref_data,
                ref_annotation_tuple=("cell_type", "astrocyte"),
                plot_volcano=True,
                method='wilcoxon'
            )
    """

    comb_col_name = "combined_annotation_column"

    # extract annotation information
    if data_annotation_tuple is None:
        annotation_key = None
        annotation_name = None
        assert ref_annotation_tuple is None
    else:
        annotation_key = data_annotation_tuple[0]
        annotation_name = data_annotation_tuple[1]

    # extract information from reference tuple (added if re_annotation_tuple is None)
    if ref_annotation_tuple == "rest":
        assert ref_data is None, "If `reference_tuple` is 'rest', `reference_data` must be None."
        ref_annotation_key = None
        reference_name = "rest"
    elif isinstance(ref_annotation_tuple, tuple) and (len(ref_annotation_tuple) == 2):
        ref_annotation_key = ref_annotation_tuple[0]
        reference_name = ref_annotation_tuple[1]
    elif ref_annotation_tuple == None:
        ref_annotation_key = None
        reference_name = None
    else:
        raise ValueError("`reference_tuple` is neither 'rest' nor a 2-tuple.")

    # check if the annotations in data and reference nceed to be checked
    if annotation_key is not None:
        _check_assignment(data=data, key=annotation_key, force_assignment=force_assignment, modality="annotations")

    # check if the reference needs to be checked
    check_reference_during_substitution = True if ref_data is None else False

    if region_tuple is not None:
        assert ref_data is None, "If `region_tuple` is given, `reference_data` must be None."

        region_key = region_tuple[0]
        region_name = region_tuple[1]

        # assign region
        _check_assignment(data=data, key=region_key, force_assignment=force_assignment, modality="regions")

    # extract main anndata
    adata1 = data.cells.matrix.copy()

    if region_tuple is not None:
        # select only one region
        region_mask = [region_name in elem for elem in adata1.obsm["regions"][region_key]]
        assert np.any(region_mask), f"Region '{region_name}' not found in key '{region_key}'."

        print(f"Restrict analysis to region '{region_name}' from key '{region_key}'.", flush=True)
        adata1 = adata1[region_mask].copy()

    if data_annotation_tuple is not None:
        col_with_id = adata1.obsm["annotations"].apply(
            func=lambda row: _substitution_func(
                row=row,
                annotation_key=annotation_key,
                annotation_name=annotation_name,
                reference_name=reference_name,
                check_reference=check_reference_during_substitution,
                ignore_duplicate_assignments=ignore_duplicate_assignments
                ), axis=1
            )

        # check that the annotation_name exists inside the column
        assert np.any(col_with_id == annotation_name), f"annotation_name '{annotation_name}' not found under annotation_key '{annotation_key}'."

        # mark the annotations with 1 or 2 depending if it is adata1 or adata2
        if ref_data is not None:
            # add a 1- in front of the annotation to differentiate it later from the reference data
            col_with_id = col_with_id.apply(func=lambda x: f"1-{x}")

        # add the column to obs
        adata1.obs[comb_col_name] = col_with_id

    if ref_data is not None:
        # process reference_data if it is not None
        if ref_annotation_tuple is None and data_annotation_tuple is not None:
            ref_annotation_tuple = data_annotation_tuple

        if ref_annotation_key is not None:
            _check_assignment(data=ref_data, key=ref_annotation_key, force_assignment=force_assignment, modality="annotations")

            # extract reference anndata
        adata2 = ref_data.cells.matrix.copy()
        # repeat the same as for adata1 for adata2
        if ref_annotation_tuple is not None:
            col_with_id_ref = adata2.obsm["annotations"].apply(
                func=lambda row: _substitution_func(
                    row=row,
                    annotation_key=ref_annotation_key,
                    annotation_name=reference_name,
                    reference_name=None,
                    check_reference=check_reference_during_substitution,
                    ignore_duplicate_assignments=ignore_duplicate_assignments
                    ), axis=1
                )
            col_with_id_ref = col_with_id_ref.apply(func=lambda x: f"2-{x}")

            # check that the reference_name exists inside the column
            assert np.any(col_with_id_ref == f"2-{reference_name}"), f"reference_name '{reference_name}' not found under reference_key '{ref_annotation_key}'."

            # add column to obs
            adata2.obs[comb_col_name] = col_with_id_ref

        if data_annotation_tuple is None and ref_annotation_tuple is None:
            adata1.obs[comb_col_name]="adata1"
            adata2.obs[comb_col_name]="adata2"

        # combine anndatas
        adata_combined = anndata.concat([adata1, adata2])

        # create settings for rank_genes_groups
        if data_annotation_tuple is not None and ref_annotation_tuple is not None:
            rgg_groups = [f"1-{annotation_name}"]
            rgg_reference = f"2-{reference_name}"

        if title is None:
            # create plot title for later
            if data_annotation_tuple is not None and ref_annotation_tuple is not None:
                plot_title = f"'{annotation_name}' in {data.sample_id} vs. '{reference_name}' in {ref_data.sample_id}"
            if data_annotation_tuple is None and ref_annotation_tuple is None:
                plot_title = f"{data.sample_id} vs.{ref_data.sample_id}"
        else:
            plot_title = title
    else:
        adata_combined = adata1
        rgg_groups = [annotation_name]
        rgg_reference = reference_name

        if title is None:
            plot_title = f"'{annotation_name}' in {data.sample_id} vs. '{reference_name}' in {data.sample_id}"
        else:
            plot_title = title

    if obs_tuple is not None:
        # filter for observation value
        adata_combined = adata_combined[adata_combined.obs[obs_tuple[0]] == obs_tuple[1]].copy()
        if data_annotation_tuple is None and ref_annotation_tuple is None:
            rgg_groups = list(adata_combined.obs[comb_col_name].unique())
            #rgg_reference=ref_data.sample_id
            print(f"Calculate differentially expressed genes with Scanpy's `rank_genes_groups` using '{method}'.")
            sc.tl.rank_genes_groups(adata=adata_combined,
                                    groupby=comb_col_name,
                                    method=method)
             # create dataframe from results
            res_dict = create_deg_dataframe(
                adata=adata_combined, groups=rgg_groups)
            df = res_dict[rgg_groups[0]]

    if data_annotation_tuple is not None and ref_annotation_tuple is not None:
        # add column to .obs for its use in rank_genes_groups()
        adata_combined.obs = adata_combined.obs.filter([comb_col_name]) # empty obs
        #adata_combined.obs[comb_col_name] = adata_combined.obsm["annotations"][comb_col_name]
        print(f"Calculate differentially expressed genes with Scanpy's `rank_genes_groups` using '{method}'.")
        sc.tl.rank_genes_groups(adata=adata_combined,
                                groupby=comb_col_name,
                                groups=rgg_groups,
                                reference=rgg_reference,
                                method=method,
                                )

        # create dataframe from results
        res_dict = create_deg_dataframe(
            adata=adata_combined, groups=rgg_groups)
        df = res_dict[rgg_groups[0]]

    if plot_volcano:
        volcano_plot(
            data=df,
            title=plot_title,
            savepath = savepath,
            save_only = save_only,
            dpi_save = dpi_save,
            **kwargs
            )
    else:
        return {
            "results": df,
            "params": adata_combined.uns["rank_genes_groups"]["params"]
        }
