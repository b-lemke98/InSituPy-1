from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any, Literal
import os
from shapely import Point
from .utils import convert_to_list
import numpy as np
import pandas as pd
from .utils import textformat as tf

# force geopandas to use shapely. Default in future versions of geopandas.
os.environ['USE_PYGEOS'] = '0' 

import geopandas
from shapely import Polygon

def read_qupath_annotation(file: Union[str, os.PathLike, Path], 
                           use_geopandas: bool = True
                           ) -> pd.DataFrame:
    """
    Reads QuPath annotation data from a file and processes it into a pandas DataFrame.
    
    Args:
        file (Union[str, os.PathLike, Path]): The path to the QuPath annotation file.
        use_geopandas (bool, optional): If True, use geopandas to read the file. 
                                        Defaults to True.
        
    Returns:
        pd.DataFrame: A DataFrame containing processed QuPath annotation data.
        
    Raises:
        Exception: If the file reading or processing fails.
    """
    # read dataframe
    df = geopandas.read_file(file)

    # flatten classification
    df["name"] = [elem["name"] for elem in df["classification"]]
    df["color"] = [elem["color"] for elem in df["classification"]]
    
    # remove redundant columns
    df = df.drop(["classification"], axis=1)
    
    return df

# def get_annotations_from_adata(adata, uns_key):
#     # extract datafram
#     df = adata.uns[uns_key]

#     # reshape coordinates
#     df["geometry"] = [tuple([x,y]) for x,y in zip(df["x"], df["y"])]
#     df_new = df.groupby("id").head(1).copy()
#     df_new = df_new.set_index("id")
#     df_new["geometry"] = df.groupby("id")["geometry"].agg(lambda x: Polygon(x.tolist()))

#     return df_new

def annotate(self, 
            annotation_labels: str = "all",
            add_annotation_masks: bool = False
            ):
    '''
    Function to assign the annotations to the anndata object in XeniumData.matrix.
    Annotation information is added to the DataFrame in `.obs`.
    '''
    # assert that prerequisites are met
    assert hasattr(self, "matrix"), "No .matrix attribute available. Run `read_matrix()`."
    assert hasattr(self, "annotations"), "No .matrix attribute available. Run `read_matrix()`."
    
    if annotation_labels == "all":
        annotation_labels = self.annotations.labels
        
    # make sure annotation labels are a list
    annotation_labels = convert_to_list(annotation_labels)
    
    # convert coordinates into shapely Point objects
    points = [Point(elem) for elem in self.matrix.obsm["spatial"]]

    # iterate through annotation labels
    for annotation_label in annotation_labels:
        # extract pandas dataframe of current label
        annot = getattr(self.annotations, annotation_label)
        
        # get index of annotation in AnnotationData
        label_idx = self.annotations.labels.index(annotation_label)
        
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
            # add results to adata.obs
            #self.matrix.obs[f"annotation-{annotation_label}__{n.replace(' ', '_')}"] = in_poly_res
            
            # collect results
            df[n] = in_poly_res
            
        # convert into pandas dataframe
        df = pd.DataFrame(df)
        df.index = self.matrix.obs_names
        
        # create annotation from annotation masks
        df[f"annotation-{annotation_label}"] = [" & ".join(annot_names[row.values]) if np.any(row.values) else np.nan for i, row in df.iterrows()]
        
        if add_annotation_masks:
            self.matrix.obs = pd.merge(left=self.matrix.obs, right=df, left_index=True, right_index=True)
        else:
            self.matrix.obs = pd.merge(left=self.matrix.obs, right=df.iloc[:, -1], left_index=True, right_index=True)
            
        # save that the current label was analyzed
        self.annotations.analyzed[label_idx] = tf.TICK