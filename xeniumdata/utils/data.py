from typing import Optional, Tuple, Union, List, Dict, Any, Literal
from pathlib import Path
import os
import pandas as pd
#from dask_image.imread import imread
from tifffile import imread
from .utils import textformat as tf
from .utils import convert_to_list, load_pyramid
from parse import *

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
        #self.n_classes.append(len(dataframe.name.unique()))

class ImageData:
    '''
    Object to read and load images.
    '''
    def __init__(self, 
                 path: Union[str, os.PathLike, Path], 
                 img_files: List[str], 
                 img_names: List[str], 
                 ):
        
        self.files = img_files
        
        self.metadata = {}
        for n, f in zip(img_names, self.files):
            # load images
            store = imread(path / f, aszarr=True)
            pyramid = load_pyramid(store)
            
            # set attribute
            setattr(self, n, pyramid) 
            
            # add metadata
            self.metadata[n] = {}
            self.metadata[n]["shape"] = pyramid[0].shape # store shape
            self.metadata[n]["subresolutions"] = len(pyramid) - 1 # store number of subresolutions of pyramid
            
            # check whether the image is RGB or not
            if len(pyramid[0].shape) == 3:
                self.metadata[n]["rgb"] = True
            elif len(pyramid[0].shape) == 2:
                self.metadata[n]["rgb"] = False
            else:
                raise ValueError(f"Unknown image shape: {pyramid[0].shape}")
            
            # get image contrast limits
            if self.metadata[n]["rgb"]:
                self.metadata[n]["contrast_limits"] = [0, 255]
            else:
                self.metadata[n]["contrast_limits"] = [0, int(pyramid[0].max())]
            
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
            
class BoundariesData:
    '''
    Object to read and load boundaries of cells and nuclei.
    '''
    def __init__(self, 
                 path: Union[str, os.PathLike, Path], 
                 cell_boundaries_file: str = "cell_boundaries.parquet",
                 nucleus_boundaries_file: str = "nucleus_boundaries.parquet"
                 ):
        # generate paths
        cellbound_path = path / "cell_boundaries.parquet"
        nucbound_path = path / "nucleus_boundaries.parquet"
        
        # load data and add to object
        setattr(self, "cells", pd.read_parquet(cellbound_path))
        setattr(self, "nuclei", pd.read_parquet(nucbound_path))
        
    def __repr__(self):
        repr_strings = [f"{tf.Bold+a+tf.ResetAll}" for a in ["cells", "nuclei"]]
        s = "\n".join(repr_strings)
        repr = f"{tf.Purple+tf.Bold}boundaries{tf.ResetAll}\n{s}"
        return repr
