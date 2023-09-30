from typing import Optional, Tuple, Union, List, Dict, Any, Literal
from pathlib import Path
import os
import pandas as pd
from dask_image.imread import imread
from .utils import textformat as tf
from parse import *

class AnnotationData:
    '''
    Object to store annotations.
    '''
    def __init__(self):
        self.names = []
        self.n_annotations = []
        self.classes = []
        #self.n_classes = []
        
    def __repr__(self):
        if len(self.names) > 0:
            repr_strings = [f"{tf.Bold}{a}:{tf.ResetAll}\t{b} annotations, {len(c)} classes {*c,}" for a,b,c in zip(self.names, 
                                                                                                    self.n_annotations, 
                                                                                                    self.classes
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
        self.names.append(name)
        self.n_annotations.append(len(dataframe))
        self.classes.append(dataframe.name.unique())
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
        
        self.img_files = img_files
        self.img_names = img_names
        
        self.img_shapes = []
        for n, f in zip(self.img_names, self.img_files):
            # load images
            img = imread(path / f)[0]
            setattr(self, n, img)
            self.img_shapes.append(img.shape)
        
    def __repr__(self):
        repr_strings = [f"{tf.Bold}{a}:{tf.ResetAll}\t{b}" for a,b in zip(self.img_names, self.img_shapes)]
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
        which = [which] if isinstance(which, str) else list(which)
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
