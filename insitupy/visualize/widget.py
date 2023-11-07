import napari
import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any, Literal
from ..palettes import CustomPalettes
from scipy.sparse import issparse
from magicgui import magicgui
from magicgui.widgets import FunctionGui
from pandas.api.types import is_numeric_dtype

def initialize_point_widgets(
    matrix,
    pixel_size: float
    ) -> Tuple[FunctionGui, FunctionGui]:
        
    # get available genes
    genes = matrix.var_names.tolist()
    obses = matrix.obs.columns.tolist()
    
    # get point coordinates
    points = np.flip(matrix.obsm["spatial"].copy(), axis=1) * pixel_size # switch x and y (napari uses [row,column])
    
    # get expression matrix
    if issparse(matrix.X):
        X = matrix.X.toarray()
    else:
        X = matrix.X    
    
    @magicgui(
            call_button='Add',
            gene={'choices': genes},
            )
    def add_genes(gene=None) -> napari.types.LayerDataTuple:
        # get expression values
        geneid = matrix.var_names.get_loc(gene)
        color_value = X[:, geneid]
        
        # set color settings for continuous data
        color_map = "viridis"
        color_cycle = None
        climits = [0, np.percentile(color_value, 95)]
        
        # generate point layer
        layer = (
            points, 
            {
                'name': gene,
                'properties': {"color_value": color_value},
                'symbol': 'o',
                'size': 30 * pixel_size,
                'face_color': "color_value",
                'face_color_cycle': color_cycle,
                'face_colormap': color_map,
                'face_contrast_limits': climits,
                'opacity': 1,
                'visible': True,
                'edge_width': 0
                }, 
            'points'
            )
        return layer
    
    @magicgui(
        call_button='Add',
        observation={'choices': obses}
        )
    def add_observations(observation=None) -> napari.types.LayerDataTuple:
        # get observation values
        color_value = matrix.obs[observation]
        
        # check if the data should be plotted categorical or continous
        if is_numeric_dtype(color_value):
            categorical = False # if the data is numeric it should be plotted continous
        else:
            categorical = True # if the data is not numeric it should be plotted categorically
            
        if categorical:
            # get color cycle for categorical data
            palettes = CustomPalettes()
            color_cycle = getattr(palettes, "tab20_mod").colors
            color_map = None
            climits = None
        else:
            color_map = "viridis"
            color_cycle = None
            climits = [0, np.percentile(color_value, 95)]
        
        # generate point layer
        layer = (
            points, 
            {
                'name': observation,
                'properties': {"color_value": color_value.values},
                'symbol': 'o',
                'size': 30 * pixel_size,
                'face_color': "color_value",
                'face_color_cycle': color_cycle,
                'face_colormap': color_map,
                'face_contrast_limits': climits,
                'opacity': 1,
                'visible': True,
                'edge_width': 0
                }, 
            'points'
            )
        return layer
    
    return add_genes, add_observations