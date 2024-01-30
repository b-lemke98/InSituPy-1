import os
import warnings
from copy import deepcopy
from os.path import relpath
from pathlib import Path
from typing import List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xmltodict
from anndata import AnnData
from parse import *
from shapely.geometry.multipolygon import MultiPolygon
from shapely import Polygon, affinity
from tifffile import TiffFile, imread

from insitupy import __version__

from .._exceptions import InvalidFileTypeError
from ..utils.geo import parse_geopandas
from ..utils.io import check_overwrite_and_remove_if_true, load_pyramid, write_dict_to_json
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
                 assert_uniqueness: Optional[bool] = None
                 ) -> None:
        self.metadata = {}
        
        if assert_uniqueness is None:
            assert_uniqueness = self.default_assert_uniqueness
        
        if files is not None:
            for key, file in zip(keys, files):
                # read annotation and store in dictionary
                self.add_annotation(data=file, 
                                    key=key, 
                                    pixel_size=pixel_size,
                                    assert_uniqueness=assert_uniqueness
                                    )
                
    def __repr__(self):
        if len(self.metadata) > 0:
            repr_strings = [
                f'{tf.Bold}{l}:{tf.ResetAll}\t{m[f"n_{self.shape_name}"]} {self.shape_name}, {len(m["classes"])} classes {*m["classes"],} {m["analyzed"]}' for l, m in self.metadata.items()
                ]
            
            s = "\n".join(repr_strings)
        else:
            s = ""
        repr = f"{self.repr_color}{tf.Bold}{self.shape_name}{tf.ResetAll}\n{s}"
        return repr   
    
    def _update_metadata(self, 
                         key: str,
                         analyzed: bool
                         ):
        # retrieve dataframe
        annot_df = getattr(self, key)
        
        # record metadata information
        self.metadata[key][f"n_{self.shape_name}"] = len(annot_df)  # number of annotations
        self.metadata[key]["classes"] = annot_df['name'].unique().tolist()  # annotation classes
        self.metadata[key]["analyzed"] = tf.Tick if analyzed else ""  # whether this annotation has been used in the annotate() function
        
            
    def add_annotation(self,
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
            # concatenate the new and old dataframe
            annot_df = getattr(self, key)

            # concatenate old and new annoation dataframe
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
                if len(annot_df.index.unique()) != len(annot_df.name.unique()):
                    warnings.warn(message=f"Names of {self.shape_name} for key '{key}' were not unique. Key was skipped.")
                    add = False
                else:
                    if verbose:
                        print(f"Names of {self.shape_name} for key '{key}' are unique.")
            
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
    def __init__(self):
        self.labels = []
        
    def __repr__(self):
        if len(self.labels) == 0:
            repr = f"Empty BoundariesData object"
        else:
            repr = f"BoundariesData object with {len(self.labels)} entries:"
            for l in self.labels:
                repr += f"\n{tf.SPACER+tf.Bold+l+tf.ResetAll}"
        return repr
    
    def read_boundaries(self,
        files: Union[str, os.PathLike, Path, List],
        labels: Optional[Union[str, List[str]]] = None,
    ):
        
        # generate dataframes
        dataframes = {}
        for n, f in zip(labels, files):
            # check the file suffix
            if not f.suffix == ".parquet":
                InvalidFileTypeError(allowed_types=[".parquet"], received_type=f.suffix)
            
            # load dataframe
            df = pd.read_parquet(f)

            # decode columns
            df = df.apply(lambda x: decode_robust_series(x), axis=0)

            # collect dataframe
            dataframes[n] = df
            
        # add dictionary with boundaries to BoundariesData object
        self.add_boundaries(dataframes=dataframes)
        
    def add_boundaries(self,
                       dataframes: Optional[Union[dict, List[str]]] = None,
                       labels: Optional[List[str]] = [],
                       overwrite: bool = False
                       ):
        if dataframes is not None:
            if isinstance(dataframes, dict):
                # extract keys from dictionary
                labels = dataframes.keys()
                dataframes = dataframes.values()
            elif isinstance(dataframes, list):
                if labels is None:
                    raise ValueError("Argument 'labels' is None. If 'dataframes' is a list, 'labels' is required to be a list, too.")
            else:
                raise ValueError(f"Argument 'dataframes' has unknown file type ({type(dataframes)}). Expected to be a list or dictionary.")
            
            for l, df in zip(labels, dataframes):
                if l not in self.labels or overwrite:
                    # add to object
                    setattr(self, l, df)
                    self.labels.append(l)
                else:
                    raise KeyError(f"Label '{l}' exists already in BoundariesData object. To overwrite, set 'overwrite' argument to True.")
                
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
        
        for n in ["cells", "nuclei"]:
            # get dataframe
            df = getattr(self, n)
            
            # filter dataframe
            df.loc[df["cell_id"].isin(cell_ids), :]
            
            # re-center to 0
            df["vertex_x"] -= xlim[0]
            df["vertex_y"] -= ylim[0]
            
            # add to object
            setattr(self, n, df)
            
                
    def convert_to_shapely_objects(self):
        for n in self.labels:
            print(f"Converting `{n}` to GeoPandas DataFrame with shapely objects.")
            # retrief dataframe with boundary coordinates
            df = getattr(self, n)
            
            # convert xy coordinates into shapely Point objects
            df["geometry"] = gpd.points_from_xy(df["vertex_x"], df["vertex_y"])
            del df["vertex_x"], df["vertex_y"]

            # convert points into polygon objects per cell id
            df = df.groupby("cell_id")['geometry'].apply(lambda x: Polygon(x.tolist()))
            df.index = decode_robust_series(df.index)  # convert byte strings in index
            
            # add to object
            setattr(self, n, pd.DataFrame(df))

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
             overwrite: bool = False
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
            for n in ["cellular", "nuclear"]:
                bound_df = getattr(boundaries, n)
                bound_file = bound_path / f"{n}.parquet"
                bound_df.to_parquet(bound_file)
                metadata["boundaries"][n] = Path(relpath(bound_file, path)).as_posix()
                
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
        cell_ids = self.matrix.obs_names
        
        try:
            boundaries = self.boundaries
        except AttributeError:
            print('No `boundaries` attribute found in CellData found.')
            pass
        else:
            for n in ["cellular", "nuclear"]:
                # get dataframe
                df = getattr(boundaries, n)
                
                # filter dataframe
                df.loc[df["cell_id"].isin(cell_ids), :]
                
                # add to object
                setattr(self.boundaries, n, df)
            
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
            for n in ["cellular", "nuclear"]:
                # get dataframe
                df = getattr(boundaries, n)
                
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
            
            # load images
            store = imread(path / f, aszarr=True)
            pyramid = load_pyramid(store)
            
            # set attribute and add names to object
            setattr(self, n, pyramid)
            self.names.append(n)
            
            # add metadata
            self.metadata[n] = {}
            self.metadata[n]["file"] = f # store file information
            self.metadata[n]["shape"] = pyramid[0].shape # store shape
            self.metadata[n]["subresolutions"] = len(pyramid) - 1 # store number of subresolutions of pyramid
            
            # read ome metadata
            with TiffFile(path / f) as tif:
                axes = tif.series[0].axes # get axes
                ome_meta = tif.ome_metadata # read OME metadata
                
            self.metadata[n]["axes"] = axes
            self.metadata[n]["OME"] = xmltodict.parse(ome_meta, attr_prefix="")["OME"] # convert XML to dict
            
            # check whether the image is RGB or not
            if len(pyramid[0].shape) == 3:
                self.metadata[n]["rgb"] = True
            elif len(pyramid[0].shape) == 2:
                self.metadata[n]["rgb"] = False
            else:
                raise ValueError(f"Unknown image shape: {pyramid[0].shape}")
            
            # get image contrast limits
            if self.metadata[n]["rgb"]:
                self.metadata[n]["contrast_limits"] = (0, 255)
            else:
                self.metadata[n]["contrast_limits"] = (0, int(pyramid[0].max()))
                
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
            pyramid = getattr(self, n)
            
            # extract pixel size
            pixel_size = self.metadata[n]['pixel_size']
            
            # get scale factors between the different pyramid levels
            scale_factors = [1] + [pyramid[i].shape[0] / pyramid[i+1].shape[0] for i in range(len(pyramid)-1)]
            
            cropped_pyramid = []
            xlim_scaled = (xlim[0] / pixel_size, xlim[1] / pixel_size) # convert to metric unit
            ylim_scaled = (ylim[0] / pixel_size, ylim[1] / pixel_size) # convert to metric unit
            for img, sf in zip(pyramid, scale_factors):
                # do cropping while taking the scale factor into account
                # scale the x and y limits
                xlim_scaled = (int(xlim_scaled[0] / sf), int(xlim_scaled[1] / sf))
                ylim_scaled = (int(ylim_scaled[0] / sf), int(ylim_scaled[1] / sf))
                
                # do the cropping
                cropped_pyramid.append(img[ylim_scaled[0]:ylim_scaled[1], xlim_scaled[0]:xlim_scaled[1]])
                
            # save cropping properties in metadata
            self.metadata[n]["cropping_xlim"] = xlim
            self.metadata[n]["cropping_ylim"] = ylim
            self.metadata[n]["shape"] = cropped_pyramid[0].shape
                
            # add cropped pyramid to object
            setattr(self, n, cropped_pyramid)
        