import napari
import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any, Literal
from ..palettes import CustomPalettes
from scipy.sparse import issparse
from magicgui import magicgui
from magicgui.widgets import FunctionGui

def initialize_widgets(
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
        expr = X[:, geneid]
        
        # set color settings for continuous data
        color_map = "viridis"
        color_cycle = None
        climits = [0, np.percentile(expr, 95)]
        
        # generate point layer
        layer = (
            points, 
            {
                'name': gene,
                'properties': {"expr": expr},
                'symbol': 'o',
                'size': 30 * pixel_size,
                'face_color': "expr",
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
        expr = matrix.obs[observation].values
        
        # get color cycle for categorical data
        palettes = CustomPalettes()
        color_cycle = getattr(palettes, "tab20_mod").colors
        color_map = None
        climits = None
        
        # generate point layer
        layer = (
            points, 
            {
                'name': observation,
                'properties': {"expr": expr},
                'symbol': 'o',
                'size': 30 * pixel_size,
                'face_color': "expr",
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