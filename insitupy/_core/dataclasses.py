import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import geopandas as gpd
import pandas as pd
import xmltodict
from parse import *
from shapely import Polygon
from tifffile import TiffFile, imread

from ..utils.geo import parse_geopandas
from ..utils.utils import convert_to_list, decode_robust_series, load_pyramid
from ..utils.utils import textformat as tf


class AnnotationData:
    '''
    Object to store annotations.
    '''
    def __init__(self,
                 annot_files: Optional[List[Union[str, os.PathLike, Path]]] = None, 
                 annot_labels: Optional[List[str]] = None
                 ) -> None:
        self.metadata = {}
        # self.n_annotations = []
        # self.classes = []
        # self.analyzed = []
        
        if annot_files is not None:
            for label, file in zip(annot_labels, annot_files):
                # read annotation and store in dictionary
                self.add_annotation(data=file, label=label)
            
    def __repr__(self):
        if len(self.metadata) > 0:
            # repr_strings = [f"{tf.Bold}{a}:{tf.ResetAll}\t{b} annotations, {len(c)} classes {*c,} {d}" for a,b,c,d in zip(self.labels, 
            #                                                                                         self.n_annotations, 
            #                                                                                         self.classes,
            #                                                                                         self.analyzed
            #                                                                                         )]
            
            repr_strings = [
                f'{tf.Bold}{l}:{tf.ResetAll}\t{m["n_annotations"]} annotations, {len(m["classes"])} classes {*m["classes"],} {m["analyzed"]}' for l, m in self.metadata.items()
                ]
            
            s = "\n".join(repr_strings)
        else:
            s = ""
        repr = f"{tf.Cyan+tf.Bold}annotations{tf.ResetAll}\n{s}"
        return repr
    
    def _update_metadata(self, 
                         label: str,
                         analyzed: bool
                         ):
        # retrieve dataframe
        annot_df = getattr(self, label)
        
        # record metadata information
        self.metadata[label]["n_annotations"] = len(annot_df)  # number of annotations
        self.metadata[label]["classes"] = annot_df['name'].unique().tolist()  # annotation classes
        self.metadata[label]["analyzed"] = tf.Tick if analyzed else ""  # whether this annotation has been used in the annotate() function
        
            
    def add_annotation(self,
                       data: Union[gpd.GeoDataFrame, pd.DataFrame, dict, 
                                   str, os.PathLike, Path],
                       label: str,
                       verbose: bool = False
                       ):
        # parse geopandas data from dataframe or file
        new_df = parse_geopandas(data)

        if not hasattr(self, label):
            # if label does not exist yet the new df is the whole annotation dataframe
            annot_df = new_df
        
            # add new entry to metadata
            self.metadata[label] = {}
            
            # collect additional variables for reporting
            new_annotations_added = True # dataframe will be added later
            existing_str = ""
            old_n = 0
            new_n = len(annot_df)
        else:
            # concatenate the new and old dataframe
            annot_df = getattr(self, label)

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
            # add dataframe to AnnotationData object
            setattr(self, label, annot_df)
            
            # update metadata
            self._update_metadata(label=label, analyzed=False)
            
            if verbose:
                # report
                print(f"Added {new_n - old_n} new annotations to {existing_str}label '{label}'")
            
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
                 files: List[Union[str, os.PathLike, Path]],
                 labels: List[str],
                 pixel_size: Optional[float] = None
                 ):
        for n, f in zip(labels, files):
            # generate paths
            filepath = Path(os.path.normpath(path / f))
            
            # load dataframe
            bounddf = pd.read_parquet(filepath)
            
            # decode columns
            bounddf = bounddf.apply(lambda x: decode_robust_series(x), axis=0)
            
            if pixel_size is not None:
                # convert coordinates into pixel coordinates
                coord_cols = ["vertex_x", "vertex_y"]
                bounddf[coord_cols] = bounddf[coord_cols].apply(lambda x: x / pixel_size)
                            
            # add to object
            setattr(self, n, bounddf)
        
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