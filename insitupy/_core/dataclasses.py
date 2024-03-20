import os
import warnings
from copy import deepcopy
from numbers import Number
from os.path import relpath
from pathlib import Path
from typing import List, Optional, Tuple, Union

import dask
import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import xmltodict
import zarr
from anndata import AnnData
from parse import *
from shapely import Polygon, affinity
from shapely.geometry.multipolygon import MultiPolygon
from tifffile import TiffFile

from insitupy import __version__
from insitupy.utils.utils import (convert_int_to_xenium_hex,
                                  convert_xenium_hex_to_int)

from .._exceptions import InvalidDataTypeError, InvalidFileTypeError
from ..image.io import read_ome_tiff, write_ome_tiff
from ..image.utils import create_img_pyramid, crop_dask_array_or_pyramid
from ..utils.geo import parse_geopandas, write_qupath_geojson
from ..utils.io import check_overwrite_and_remove_if_true, write_dict_to_json
from ..utils.utils import convert_to_list, decode_robust_series
from ..utils.utils import textformat as tf
from ._mixins import DeepCopyMixin


class ShapesData(DeepCopyMixin):
    '''
    Object to store annotations.
    '''
    default_assert_uniqueness = False
    shape_name = "shapes"
    repr_color = tf.Cyan
    def __init__(self,
                 files: Optional[List[Union[str, os.PathLike, Path]]] = None, 
                 keys: Optional[List[str]] = None,
                 pixel_size: float = 1,
                 assert_uniqueness: Optional[bool] = None,
                 # shape_name: Optional[str] = None
                 ) -> None:
        
        # create dictionary for metadata
        self.metadata = {}
        
        if files is not None:
            # make sure files and keys are a list
            files = convert_to_list(files)
            keys = convert_to_list(keys)
            assert len(files) == len(keys), "Number of files does not match number of keys."
                
            if assert_uniqueness is None:
                assert_uniqueness = self.default_assert_uniqueness
            
            if files is not None:
                for key, file in zip(keys, files):
                    # read annotation and store in dictionary
                    self.add_shapes(data=file, 
                                        key=key, 
                                        pixel_size=pixel_size,
                                        assert_uniqueness=assert_uniqueness
                                        )
                
    def __repr__(self):
        if len(self.metadata) > 0:
            repr_strings = []
            for l, m in self.metadata.items():
                # add ' to classes
                classes = [f"'{elem}'" for elem in m["classes"]]
                lc = len(classes)
                
                # create string
                r = (
                    f'{tf.Bold}{l}:{tf.ResetAll}\t{m[f"n_{self.shape_name}"]} ' 
                    f'{self.shape_name}, {lc} '
                    f'{"classes" if lc>1 else "class"} '
                )
                if lc < 10:
                    r += f'({",".join(classes)}) {m["analyzed"]}'
                repr_strings.append(r)
            # repr_strings = [
            #     #f'{tf.Bold}{l}:{tf.ResetAll}\t{m[f"n_{self.shape_name}"]} {self.shape_name}, {len(m["classes"])} classes {*m["classes"],} {m["analyzed"]}' for l, m in self.metadata.items()
            #     f'{tf.Bold}{l}:{tf.ResetAll}\t{m[f"n_{self.shape_name}"]} {self.shape_name}, {len(m["classes"])} {"classes" if len(m["classes"])>1 else "class"} ({",".join(m["classes"])}) {m["analyzed"]}' for l, m in self.metadata.items()
            #     ]
            
            s = "\n".join(repr_strings)
        else:
            s = ""
        repr = f"{self.repr_color}{tf.Bold}{self.shape_name}{tf.ResetAll}\n{s}"
        return repr
    
    def _check_uniqueness(self,
                          dataframe: Optional[gpd.GeoDataFrame] = None,
                          key: Optional[str] = None,
                          verbose: bool = True
                          ) -> bool:
        
        if dataframe is None:
            annot_df = getattr(self, key)
        else:
            annot_df = dataframe
        
        if len(annot_df.index.unique()) != len(annot_df.name.unique()):
            warnings.warn(message=f"Names of {self.shape_name} for key '{key}' were not unique. Key was skipped.")
            return False
        else:
            if verbose:
                print(f"Names of {self.shape_name} for key '{key}' are unique.")
            return True
    
    def _update_metadata(self, 
                         key: str,
                         analyzed: bool
                         ):
        # retrieve dataframe
        annot_df = getattr(self, key)
        
        # record metadata information
        self.metadata[key][f"n_{self.shape_name}"] = len(annot_df)  # number of annotations
        
        try:        
            self.metadata[key]["classes"] = annot_df['name'].unique().tolist()  # annotation classes
        except KeyError:
            self.metadata[key]["classes"] = ["unnamed"]
            
        self.metadata[key]["analyzed"] = tf.Tick if analyzed else ""  # whether this annotation has been used in the annotate() function
        
            
    def add_shapes(self,
                   data: Union[gpd.GeoDataFrame, pd.DataFrame, dict, 
                                str, os.PathLike, Path],
                   key: str,
                   pixel_size: Optional[float] = 1,
                   verbose: bool = False,
                   assert_uniqueness: bool = False
                   ):
        # parse geopandas data from dataframe or file
        new_df = parse_geopandas(data)
        
        # convert pixel coordinates to metric units
        new_df["geometry"] = new_df.geometry.scale(origin=(0,0), xfact=pixel_size, yfact=pixel_size)
        
        if not hasattr(self, key):
            # if key does not exist yet, the new df is the whole annotation dataframe
            annot_df = new_df
            
            # collect additional variables for reporting
            new_annotations_added = True # dataframe will be added later
            existing_str = ""
            old_n = 0
            new_n = len(annot_df)
        else:
            # concatenate old and new annoation dataframe
            annot_df = getattr(self, key)
            old_n = len(annot_df)
            annot_df = pd.concat([annot_df, new_df], ignore_index=False)
            
            # remove all duplicated shapes - leaving only the newly added
            annot_df = annot_df[~annot_df.index.duplicated()]
            new_n = len(annot_df)
            
            # collect additional variables for reporting
            new_annotations_added = new_n > old_n
            existing_str = "existing "
                    
        if new_annotations_added:
            add = True
            if assert_uniqueness:
                # if len(annot_df.index.unique()) != len(annot_df.name.unique()):
                #     warnings.warn(message=f"Names of {self.shape_name} for key '{key}' were not unique. Key was skipped.")
                #     add = False
                # else:
                #     if verbose:
                #         print(f"Names of {self.shape_name} for key '{key}' are unique.")
                
                # check if the shapes data for this key is unique (same number of names than indices)
                is_unique = self._check_uniqueness(dataframe=annot_df, key=key, verbose=verbose)
                
                if not is_unique:
                    add = False
            
            # check if any of the shapes are shapely MultiPolygons
            is_not_multipolygon = [not isinstance(p, MultiPolygon) for p in annot_df.geometry]
            if not np.all(is_not_multipolygon):
                annot_df = annot_df.loc[is_not_multipolygon]
                warnings.warn(f"Some {self.shape_name} were a shapely 'MultiPolygon' objects and skipped.")
            
            if add:
                # add dataframe to AnnotationData object
                setattr(self, key, annot_df)
                
                # add new entry to metadata
                self.metadata[key] = {}
                
                # update metadata
                self._update_metadata(key=key, analyzed=False)
                
                if verbose:
                    # report
                    print(f"Added {new_n - old_n} new {self.shape_name} to {existing_str}key '{key}'")
                    
    def crop(self,
             xlim, ylim
             ):
        limit_poly = Polygon([(xlim[0], ylim[0]), (xlim[1], ylim[0]), (xlim[1], ylim[1]), (xlim[0], ylim[1])])
        
        new_metadata = {}
        for i, n in enumerate(self.metadata.keys()):
            shapesdf = getattr(self, n)
            
            # select annotations that intersect with the selected area
            mask = [limit_poly.intersects(elem) for elem in shapesdf["geometry"]]
            shapesdf = shapesdf.loc[mask, :].copy()
            
            # move origin to zero after cropping
            shapesdf["geometry"] = shapesdf["geometry"].apply(affinity.translate, xoff=-xlim[0], yoff=-ylim[0])
            
            # check if there are annotations left or if it has to be deleted
            if len(shapesdf) > 0:
                # add new dataframe back to annotations object
                setattr(self, n, shapesdf)
                
                # update metadata
                new_metadata[n] = {}
                new_metadata[n][f"n_{self.shape_name}"] = len(shapesdf)
                new_metadata[n]["classes"] = shapesdf.name.unique().tolist()
                new_metadata[n]["analyzed"] = self.metadata[n]["analyzed"]  # analyzed information is just copied
            
            else:
                # delete annotations
                delattr(self, n)

        self.metadata = new_metadata

    def save(self,
             path: Union[str, os.PathLike, Path],
             overwrite: bool = False
             ):
        path = Path(path)
        
        # check if the output file should be overwritten
        check_overwrite_and_remove_if_true(path, overwrite=overwrite)
        
        # create directory
        path.mkdir(parents=True, exist_ok=True)
        
        # # create path for matrix
        # annot_path = (path / self.shape_name)
        # annot_path.mkdir(parents=True, exist_ok=True) # create directory
        
        # if metadata is not None:
        #     metadata["annotations"] = {}
        for n in self.metadata.keys():
            df = getattr(self, n)
            # annot_file = annot_path / f"{n}.parquet"
            # annot_df.to_parquet(annot_file)
            shapes_file = path / f"{n}.geojson"
            write_qupath_geojson(dataframe=df, file=shapes_file)
            
            # if metadata is not None:
            #     metadata["annotations"][n] = Path(relpath(annot_file, path)).as_posix()
            
        # save AnnotationData metadata
        shape_meta_path = path / f"metadata.json"
        write_dict_to_json(dictionary=self.metadata, file=shape_meta_path)
        
class AnnotationsData(ShapesData):
    def __init__(self,
                 files: Optional[List[Union[str, os.PathLike, Path]]] = None, 
                 keys: Optional[List[str]] = None,
                 pixel_size: float = 1
                 ) -> None:
        self.default_assert_uniqueness = False
        self.shape_name = "annotations"
        self.repr_color = tf.Cyan
        
        ShapesData.__init__(self, files, keys, pixel_size)
    
class RegionsData(ShapesData):
    def __init__(self,
                 files: Optional[List[Union[str, os.PathLike, Path]]] = None, 
                 keys: Optional[List[str]] = None,
                 pixel_size: float = 1
                 ) -> None:
        self.default_assert_uniqueness = True
        self.shape_name = "regions"
        self.repr_color = tf.Yellow
        
        ShapesData.__init__(self, files, keys, pixel_size)

class BoundariesData(DeepCopyMixin):
    '''
    Object to read and load boundaries of cells and nuclei.
    '''
    def __init__(self,
                 cell_ids: Optional[da.core.Array] = None,
                 seg_mask_value: Optional[da.core.Array] = None,
                 #pixel_size: Number = 1, # required for boundaries that are saved as masks
                 ):
        self.metadata = {}
        
        # store cell ids
        self.cell_ids = cell_ids
        self.seg_mask_value = seg_mask_value
        
    def __repr__(self):
        labels = list(self.metadata.keys())
        if len(labels) == 0:
            repr = f"Empty BoundariesData object"
        else:
            ll = len(labels)
            repr = f"BoundariesData object with {ll} {'entry' if ll == 1 else 'entries'}:"
            for l in labels:
                repr += f"\n{tf.SPACER+tf.Bold+l+tf.ResetAll}"
        return repr
        
    def add_boundaries(self,
                       data: Optional[Union[dict, List[str]]],
                       pixel_size: Number, # required for boundaries that are saved as masks
                       labels: Optional[List[str]] = [],
                       overwrite: bool = False
                       ):
        if isinstance(data, dict):
            # extract keys from dictionary
            labels = data.keys()
            data = data.values()
        elif isinstance(data, list):
            if labels is None:
                raise ValueError("Argument 'labels' is None. If 'dataframes' is a list, 'labels' is required to be a list, too.")
            else:
                # make sure labels is a list
                labels = convert_to_list(labels)
        else:
            data = convert_to_list(data)
            labels = convert_to_list(labels)
            #raise ValueError(f"Argument 'dataframes' has unknown file type ({type(data)}). Expected to be a list or dictionary.")
        
        for l, df in zip(labels, data):
            if isinstance(df, pd.DataFrame) or isinstance(df, da.core.Array) or np.all([isinstance(elem, da.core.Array) for elem in df]):             
                if l not in self.metadata or overwrite:
                    # add to object
                    setattr(self, l, df)
                    self.metadata[l] = {}
                    self.metadata[l]["pixel_size"] = pixel_size
                else:
                    raise KeyError(f"Label '{l}' exists already in BoundariesData object. To overwrite, set 'overwrite' argument to True.")
            else:
                print(f"Boundaries element `{l}` is neither a pandas DataFrame nor a Dask Array. Was not added.")
                
    def crop(self,
             cell_ids: List[str],
             xlim: Tuple[int, int],
             ylim: Tuple[int, int]
             ):
        '''
        Crop the BoundariesData object.
        '''
        # make sure cell ids are a list
        cell_ids = convert_to_list(cell_ids)
        
        for n, meta in self.metadata.items():
            # get dataframe
            data = getattr(self, n)
            
            try:
                # get pixel size
                pixel_size = meta["pixel_size"]
                
                data = crop_dask_array_or_pyramid(
                    data=data,
                    xlim=xlim,
                    ylim=ylim,
                    pixel_size=pixel_size
                )
            except InvalidDataTypeError:
                # filter dataframe
                data = data.loc[data["cell_id"].isin(cell_ids), :]
                
                # re-center to 0
                data["vertex_x"] -= xlim[0]
                data["vertex_y"] -= ylim[0]

            # add to object
            setattr(self, n, data)
            
                
    def convert_to_shapely_objects(self):
        for n in self.metadata.keys():
            print(f"Converting `{n}` to GeoPandas DataFrame with shapely objects.")
            # retrief dataframe with boundary coordinates
            df = getattr(self, n)
            
            if isinstance(df, pd.DataFrame):
                # convert xy coordinates into shapely Point objects
                df["geometry"] = gpd.points_from_xy(df["vertex_x"], df["vertex_y"])
                del df["vertex_x"], df["vertex_y"]

                # convert points into polygon objects per cell id
                df = df.groupby("cell_id")['geometry'].apply(lambda x: Polygon(x.tolist()))
                df.index = decode_robust_series(df.index)  # convert byte strings in index
                
                # add to object
                setattr(self, n, pd.DataFrame(df))
            else:
                print(f"Boundaries element `{n} was no Dataframe. Skipped.")

class CellData(DeepCopyMixin):
    '''
    Data object containing an AnnData object and a boundary object which are kept in sync.
    '''
    def __init__(self, 
               matrix: AnnData,
               boundaries: Optional[BoundariesData],
               ):
        self.matrix = matrix
        
        if boundaries is not None:
            self.boundaries = boundaries
            self.entries = ["matrix", "boundaries"]
        else:
            self.boundaries = None
            self.entries = ["matrix"]
    
    def __repr__(self):
        repr = (
            f"{tf.Bold+'matrix'+tf.ResetAll}\n"
            f"{tf.SPACER+self.matrix.__repr__()}"
        )
        
        if self.boundaries is not None:
            bound_repr = self.boundaries.__repr__()
            
            repr += f"\n{tf.Bold+'boundaries'+tf.ResetAll}\n" + tf.SPACER + bound_repr.replace("\n", f"\n{tf.SPACER}")
        return repr
    
    def copy(self):
        '''
        Function to generate a deep copy of the current object.
        '''
        
        return deepcopy(self)
    
    def save(self, 
             path: Union[str, os.PathLike, Path],
             overwrite: bool = False,
             boundaries_as_pyramid: bool = True
             ):
        
        path = Path(path)
        metadata = {}
        
        # check if the output file should be overwritten
        check_overwrite_and_remove_if_true(path, overwrite=overwrite)
        
        # create path for matrix
        mtx_path = path / "matrix"
        mtx_path.mkdir(parents=True, exist_ok=True) # create directory
        
        # write matrix to file
        mtx_file = mtx_path / "matrix.h5ad"
        self.matrix.write(mtx_file)
        metadata["matrix"] = Path(relpath(mtx_file, path)).as_posix()
        
        # save boundaries
        try:
            boundaries = self.boundaries
        except AttributeError:
            pass
        else:
            bound_path = (path / "boundaries")
            bound_path.mkdir(parents=True, exist_ok=True) # create directory
            
            metadata["boundaries"] = {}
            metadata["boundaries"]["keys"] = []
            bound_file = bound_path / f"boundaries.zarr.zip"
            with zarr.ZipStore(bound_file, mode='w') as zipstore:
                for n in boundaries.metadata.keys():
                    bound_data = getattr(boundaries, n)
                        
                    # check data
                    if isinstance(bound_data, list):
                        if not boundaries_as_pyramid:
                            bound_data = bound_data[0]
                    else:
                        if boundaries_as_pyramid:
                            # create pyramid
                            bound_data = create_img_pyramid(img=bound_data, nsubres=6)        
                        
                    
                    #if isinstance(bound_data, dask.array.core.Array):
                    if isinstance(bound_data, list):
                        for i, b in enumerate(bound_data):
                            comp = f"masks/{n}/{i}"
                            b.to_zarr(zipstore, component=comp)
                    else:
                        bound_data.to_zarr(zipstore, component=f"masks/{n}")
                        
                    # add boundaries metadata to zarr.zip
                    store = zarr.open(zipstore, mode="a")
                    store[f"masks/{n}"].attrs.put(boundaries.metadata[n])
                    
                    # save keys in insitupy metadata
                    metadata["boundaries"]["keys"].append(n)
                
                # save paths in insitupy metadata
                metadata["boundaries"]["path"] = Path(relpath(bound_file, path)).as_posix()
                
                boundaries.cell_ids.to_zarr(zipstore, component="cell_id")
                
                if boundaries.seg_mask_value is not None:
                    boundaries.seg_mask_value.to_zarr(zipstore, component="seg_mask_value")
                    
        # add more things to metadata
        metadata["version"] = __version__
        
        # save metadata
        write_dict_to_json(dictionary=metadata, file=path / ".celldata")
        
            
    def sync_cell_ids(self):
        '''
        Function to synchronize matrix and boundaries of CellData.
        
        Procedure:
        1. Select matrix cell IDs
        2. Check if all matrix cell IDs are in boundaries
            - if not all are in boundaries, throw error saying that those will also be removed
        3. Select only matrix cell IDs which are also in boundaries and filter for them
        '''
        # get cell IDs from matrix
        cell_ids_hex = self.matrix.obs_names.astype(str)
        
        # convert hex cell IDs into integers
        #cell_ids_int = [convert_xenium_hex_to_int(elem)[0] for elem in cell_ids_hex]
        
        try:
            boundaries = self.boundaries
        except AttributeError:
            print('No `boundaries` attribute found in CellData found.')
            pass
        else:
            # retrieve cell_ids of boundaries
            bound_cell_ids_int = boundaries.cell_ids[:,0].compute()
            bound_cell_ids_hex = [convert_int_to_xenium_hex(elem, dataset_suffix=1) for elem in bound_cell_ids_int]
            
            # check which ids are not in 
            not_in_matrix = boundaries.seg_mask_value[np.where([elem not in cell_ids_hex for elem in bound_cell_ids_hex])[0]].compute()
            
            for n in boundaries.metadata.keys():
                # get data
                bound_data = getattr(boundaries, n)
                
                if isinstance(bound_data, da.core.Array):
                    # set all non existent cell ids to zero
                    bound_data[da.isin(bound_data, not_in_matrix)] = 0
                elif isinstance(bound_data, pd.DataFrame):                    
                    # filter dataframe
                    bound_data = bound_data.loc[bound_data["cell_id"].astype(str).isin(cell_ids_hex), :]
                    

                else:
                    warnings.warn(f"Unknown data type for boundaries key '{n}'. Skipped synchronization of cell ids.")
                # add to object
                setattr(self.boundaries, n, bound_data)
                
    def shift(self, 
              x: Union[int, float], 
              y: Union[int, float]
              ):
        '''
        Function to shift the coordinates of both matrix and boundaries data by certain values x/y.
        '''
        
        # move origin again to 0 by subtracting the lower limits from the coordinates
        cell_coords = self.matrix.obsm['spatial'].copy()
        cell_coords[:, 0] += x
        cell_coords[:, 1] += y
        self.matrix.obsm['spatial'] = cell_coords
        
        try:
            boundaries = self.boundaries
        except AttributeError:
            print('No `boundaries` attribute found in CellData found.')
            pass
        else:
            for n in boundaries.metadata.keys():
                # get dataframe
                df = getattr(boundaries, n)
                
                if isinstance(df, pd.DataFrame):
                    # re-center to 0
                    df["vertex_x"] += x
                    df["vertex_y"] += y
                    
                    # add to object
                    setattr(self.boundaries, n, df)
        
    
class ImageData(DeepCopyMixin):
    '''
    Object to read and load images.
    '''
    def __init__(self, 
                 path: Union[str, os.PathLike, Path], 
                 img_files: List[str], 
                 img_names: List[str],
                 pixel_size: float,
                 ):
        # convert arguments to lists
        img_files = convert_to_list(img_files)
        img_names = convert_to_list(img_names)
        
        # iterate through files and load them
        self.names = []
        self.metadata = {}
        for n, f in zip(img_names, img_files):
            
            # generate full path for image
            impath = path / f
            impath = impath.resolve() # resolve relative path
            suffix = impath.name.split(".", maxsplit=1)[-1]
            
            if suffix == "zarr.zip":
                # load image from .zarr.zip
                with zarr.ZipStore(impath, mode="r") as zipstore:
                    # get components of zip store
                    components = zipstore.listdir()
    
                    if ".zarray" in components:
                        # the store is an array which can be opened
                        img = da.from_zarr(zipstore).persist()
                    else:
                        subres = [elem for elem in components if not elem.startswith(".")]
                        img = []
                        for s in subres:
                            img.append(da.from_zarr(zipstore, component=s).persist())
                    
                    # retrieve OME metadata
                    store = zarr.open(zipstore)
                    meta = store.attrs.asdict()
                    ome_meta = meta["OME"]
                    axes = meta["axes"]
                    # except KeyError:
                    #     warnings.warn("No OME metadata in `zarr.zip` file. Skipped collection of metadata.")

            elif suffix in ["ome.tif", "ome.tiff"]:
                # load image from .ome.tiff
                #img = read_ome_tiff(path=impath, levels=0)
                img = read_ome_tiff(path=impath, levels=None)
                # read ome metadata
                with TiffFile(path / f) as tif:
                    axes = tif.series[0].axes # get axes
                    ome_meta = tif.ome_metadata # read OME metadata
                    ome_meta = xmltodict.parse(ome_meta, attr_prefix="")["OME"] # convert XML to dict
                    
            else:
                raise InvalidFileTypeError(
                    allowed_types=["zarr.zip", "ome.tif", "ome.tiff"], 
                    received_type=suffix
                    )
                
            # set attribute and add names to object
            setattr(self, n, img)
            self.names.append(n)
            
            # retrieve metadata
            img_shape = img[0].shape if isinstance(img, list) else img.shape
            img_max = img[0].max() if isinstance(img, list) else img.max()
            img_max = int(img_max)

            # save metadata
            self.metadata[n] = {}
            self.metadata[n]["file"] = f # store file information
            self.metadata[n]["shape"] = img_shape  # store shape
            #self.metadata[n]["subresolutions"] = len(img) - 1 # store number of subresolutions of pyramid
            self.metadata[n]["axes"] = axes
            self.metadata[n]["OME"] = ome_meta
            
            # check whether the image is RGB or not
            if len(img_shape) == 3:
                self.metadata[n]["rgb"] = True
            elif len(img_shape) == 2:
                self.metadata[n]["rgb"] = False
            else:
                raise ValueError(f"Unknown image shape: {img_shape}")
            
            # get image contrast limits
            if self.metadata[n]["rgb"]:
                self.metadata[n]["contrast_limits"] = (0, 255)
            else:
                self.metadata[n]["contrast_limits"] = (0, img_max)
                
            # add universal pixel size to metadata
            self.metadata[n]['pixel_size'] = pixel_size
                
        
    def __repr__(self):
        repr_strings = [f"{tf.Bold}{n}:{tf.ResetAll}\t{metadata['shape']}" for n,metadata in self.metadata.items()]
        s = "\n".join(repr_strings)
        repr = f"{tf.Blue+tf.Bold}images{tf.ResetAll}\n{s}"
        return repr
    
    def load(self, 
             which: Union[List[str], str] = "all"
             ):
        '''
        Load images into memory.
        '''
        if which == "all":
            which = self.img_names
            
        # make sure which is a list
        which = convert_to_list(which)
        for n in which:
            img_loaded = getattr(self, n).compute()
            setattr(self, n, img_loaded)
            
    def crop(self,
             xlim: Tuple[int, int],
             ylim: Tuple[int, int]
             ):        
        # extract names from metadata
        names = list(self.metadata.keys())
        for n in names:
            # extract the image pyramid
            img_data = getattr(self, n)
            
            # extract pixel size
            pixel_size = self.metadata[n]['pixel_size']
            
            cropped_img_data = crop_dask_array_or_pyramid(
                data=img_data,
                xlim=xlim,
                ylim=ylim,
                pixel_size=pixel_size
            )
                
            # save cropping properties in metadata
            self.metadata[n]["cropping_xlim"] = xlim
            self.metadata[n]["cropping_ylim"] = ylim
            
            try:
                self.metadata[n]["shape"] = cropped_img_data.shape
            except AttributeError:
                self.metadata[n]["shape"] = cropped_img_data[0].shape
                
            # add cropped pyramid to object
            setattr(self, n, cropped_img_data)
        
    def save(self,
             path: Union[str, os.PathLike, Path],
             keys_to_save: Optional[str] = None,
             images_as_zarr: bool = True,
             save_pyramid: bool = True,
             return_savepaths: bool = False,
             overwrite: bool = False
             ):
        path = Path(path)
        
        if keys_to_save is None:
            keys_to_save = list(self.metadata.keys())
        else:
            keys_to_save = convert_to_list(keys_to_save)
        
        # check overwrite
        check_overwrite_and_remove_if_true(path=path, overwrite=overwrite)
        
        # create output directory
        path.mkdir(parents=True, exist_ok=True)
        
        if return_savepaths:
            savepaths = {}
        
        for n, img_metadata in self.metadata.items():
            if n in keys_to_save:
                # extract image
                img = getattr(self, n)

                if images_as_zarr:
                    # generate filename
                    filename = Path(img_metadata["file"]).name.split(".")[0] + ".zarr.zip"
                    
                    # decide whether to save as pyramid or not
                    if isinstance(img, list):
                        if not save_pyramid:
                            img = img[0]
                    else:
                        if save_pyramid:
                            # create img pyramid
                            img = create_img_pyramid(img=img, nsubres=6)
                    
                    with zarr.ZipStore(path / filename, mode="w") as zipstore:
                        # check whether to save the image as pyramid or not
                        if save_pyramid:
                            for i, im in enumerate(img):
                                im.to_zarr(zipstore, component=str(i))
                        else:
                            # save image data in zipstore without pyramid
                            img.to_zarr(zipstore)
                        
                        # open zarr store save metadata in zarr store
                        store = zarr.open(zipstore, mode="a")
                        store.attrs.put(img_metadata)
                        # for k,v in img_metadata.items():
                        #     store.attrs[k] = v
                        
                else:
                    # get file name for saving
                    filename = Path(img_metadata["file"]).name.split(".")[0] + ".ome.tif"
                    # retrieve image metadata for saving
                    photometric = 'rgb' if img_metadata['rgb'] else 'minisblack'
                    axes = img_metadata['axes']
                    
                    # retrieve OME metadata
                    ome_meta_to_retrieve = ["SignificantBits", "PhysicalSizeX", "PhysicalSizeY", "PhysicalSizeXUnit", "PhysicalSizeYUnit"]
                    pixel_meta = img_metadata["OME"]["Image"]["Pixels"]
                    selected_metadata = {key: pixel_meta[key] for key in ome_meta_to_retrieve if key in pixel_meta}
                    
                    # write images as OME-TIFF
                    write_ome_tiff(path / filename, img, 
                                photometric=photometric, axes=axes, 
                                metadata=selected_metadata, overwrite=False)
                
                if return_savepaths:
                    # collect savepaths
                    savepaths[n] = path / filename
        
        if return_savepaths:
            return savepaths
        