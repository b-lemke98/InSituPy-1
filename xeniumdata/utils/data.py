from typing import Optional, Tuple, Union, List, Dict, Any, Literal
from pathlib import Path
import os
import pandas as pd
import geopandas as gpd
from shapely import Polygon
from tifffile import imread, TiffFile
from .utils import textformat as tf
from .utils import convert_to_list, load_pyramid, decode_robust_series
from parse import *
import xmltodict
import warnings

class AnnotationData:
    '''
    Object to store annotations.
    '''
    def __init__(self):
        self.labels = []
        self.n_annotations = []
        self.classes = []
        self.analyzed = []
        #self.n_classes = []
        
    def __repr__(self):
        if len(self.labels) > 0:
            repr_strings = [f"{tf.Bold}{a}:{tf.ResetAll}\t{b} annotations, {len(c)} classes {*c,} {d}" for a,b,c,d in zip(self.labels, 
                                                                                                    self.n_annotations, 
                                                                                                    self.classes,
                                                                                                    self.analyzed
                                                                                                    )]
            s = "\n".join(repr_strings)
        else:
            s = ""
        repr = f"{tf.Cyan+tf.Bold}annotations{tf.ResetAll}\n{s}"
        return repr
        
    def add_annotation(self,
                       dataframe: pd.DataFrame,
                       name: str
                       ):
        setattr(self, name, dataframe)
        self.labels.append(name)
        self.n_annotations.append(len(dataframe))
        self.classes.append(dataframe.name.unique())
        self.analyzed.append("")

class ImageData:
    '''
    Object to read and load images.
    '''
    def __init__(self, 
                 path: Union[str, os.PathLike, Path], 
                 img_files: List[str], 
                 img_names: List[str], 
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
        names = list(self.metadata.keys())
        for n in names:
            # extract the image pyramid
            pyramid = getattr(self, n)
            
            # get scale factors between the different pyramid levels
            scale_factors = [1] + [pyramid[i].shape[0] / pyramid[i+1].shape[0] for i in range(len(pyramid)-1)]
            
            cropped_pyramid = []
            xlim_scaled = xlim
            ylim_scaled = ylim
            for img, sf in zip(pyramid, scale_factors):
                # do cropping while taking the scale factor into account
                # scale the x and y limits
                xlim_scaled = (int(xlim_scaled[0] / sf), int(xlim_scaled[1] / sf))
                ylim_scaled = (int(ylim_scaled[0] / sf), int(ylim_scaled[1] / sf))
                
                # do the cropping
                cropped_pyramid.append(img[ylim_scaled[0]:ylim_scaled[1], xlim_scaled[0]:xlim_scaled[1]])
                
            # save cropping in metadata
            self.metadata[n]["cropping_xlim"] = xlim
            self.metadata[n]["cropping_ylim"] = ylim
            self.metadata[n]["shape"] = cropped_pyramid[0].shape
                
            # add cropped pyramid to object
            setattr(self, n, cropped_pyramid)
        
            
class BoundariesData:
    '''
    Object to read and load boundaries of cells and nuclei.
    '''
    def __init__(self, 
                 path: Union[str, os.PathLike, Path], 
                 cell_boundaries_file: str = "cell_boundaries.parquet",
                 nucleus_boundaries_file: str = "nucleus_boundaries.parquet",
                 pixel_size: Optional[float] = None
                 ):
        # generate paths
        cellbound_path = path / cell_boundaries_file
        nucbound_path = path / nucleus_boundaries_file
        
        # load dataframe
        celldf = pd.read_parquet(cellbound_path)
        nucdf = pd.read_parquet(nucbound_path)
        
        # decode columns
        celldf = celldf.apply(lambda x: decode_robust_series(x), axis=0)
        nucdf = nucdf.apply(lambda x: decode_robust_series(x), axis=0)
        
        if pixel_size is not None:
            # convert coordinates into pixel coordinates
            coord_cols = ["vertex_x", "vertex_y"]
            celldf[coord_cols] = celldf[coord_cols].apply(lambda x: x / pixel_size)
            nucdf[coord_cols] = nucdf[coord_cols].apply(lambda x: x / pixel_size)
        
        # add to object
        setattr(self, "cells", celldf)
        setattr(self, "nuclei", nucdf)
        
    def __repr__(self):
        repr_strings = [f"{tf.Bold+a+tf.ResetAll}" for a in ["cells", "nuclei"]]
        s = "\n".join(repr_strings)
        repr = f"{tf.Purple+tf.Bold}boundaries{tf.ResetAll}\n{s}"
        return repr
    
    def sync_to_matrix(self,
                       cell_ids: List[str],
                       xlim: Tuple[int, int],
                       ylim: Tuple[int, int]
                       ):
        '''
        Synchronize the BoundariesData object to match the cells in self.matrix.
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
            
                
    def convert_to_geopandas(self):
        for n in ["cells", "nuclei"]:
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