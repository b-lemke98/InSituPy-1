from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any, Literal
import os

# force geopandas to use shapely. Default in future versions of geopandas.
os.environ['USE_PYGEOS'] = '0' 

import geopandas
from shapely import Polygon

def read_qupath_annotation(file: Union[str, os.PathLike, Path], 
                           use_geopandas: bool = True
                           ) -> pd.DataFrame:
    # read dataframe
    print(file)
    df = geopandas.read_file(file)

    # flatten classification
    df["name"] = [elem["name"] for elem in df["classification"]]
    df["color"] = [elem["color"] for elem in df["classification"]]
    
    # remove redundant columns
    df = df.drop(["classification"], axis=1)
    
    return df

def get_annotations_from_adata(adata, uns_key):
    # extract datafram
    df = adata.uns[uns_key]

    # reshape coordinates
    df["geometry"] = [tuple([x,y]) for x,y in zip(df["x"], df["y"])]
    df_new = df.groupby("id").head(1).copy()
    df_new = df_new.set_index("id")
    df_new["geometry"] = df.groupby("id")["geometry"].agg(lambda x: Polygon(x.tolist()))

    return df_new

