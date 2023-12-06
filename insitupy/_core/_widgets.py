from typing import Tuple

import napari
import numpy as np
from anndata import AnnData
from magicgui import magic_factory, magicgui
from magicgui.widgets import FunctionGui
from napari.types import LayerDataTuple
from pandas.api.types import is_numeric_dtype
from scipy.sparse import issparse

from ..utils.palettes import CustomPalettes


def _create_points_layer(points, 
                         color_value, 
                         name, 
                         pixel_size,
                         size_factor: int = 30,
                         opacity: float = 1,
                         visible: bool = True,
                         edge_width: float = 0
                         ) -> LayerDataTuple:
    # check if the data should be plotted categorical or continous
    if is_numeric_dtype(color_value):
        categorical = False # if the data is numeric it should be plotted continous
    else:
        categorical = True # if the data is not numeric it should be plotted categorically
        
    if categorical:
        # get color cycle for categorical data
        color_mode = "cycle"
        palettes = CustomPalettes()
        color_cycle = getattr(palettes, "tab20_mod").colors
        color_map = None
        climits = None
    else:
        color_mode = "colormap"
        color_map = "viridis"
        color_cycle = None
        climits = [0, np.percentile(color_value, 95)]
    
    # generate point layer
    layer = (
        points, 
        {
            'name': name,
            'properties': {"color_value": color_value},
            'symbol': 'o',
            'size': size_factor * pixel_size,
            'face_color': {
                "color_mode": color_mode, # workaround (see https://github.com/napari/napari/issues/6433)
                "colors": "color_value"
                },
            'face_color_cycle': color_cycle,
            'face_colormap': color_map,
            'face_contrast_limits': climits,
            'opacity': opacity,
            'visible': visible,
            'edge_width': edge_width
            }, 
        'points'
        )
    return layer

def _initialize_point_widgets(
    adata: AnnData,
    pixel_size: float
    ) -> Tuple[FunctionGui, FunctionGui]:
        
    # get available genes
    genes = adata.var_names.tolist()
    obses = adata.obs.columns.tolist()
    
    # get point coordinates
    points = np.flip(adata.obsm["spatial"].copy(), axis=1) * pixel_size # switch x and y (napari uses [row,column])
    
    # get expression matrix
    if issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X
    
    @magicgui(
            call_button='Add',
            gene={'choices': genes},
            )
    def add_genes(gene=None) -> napari.types.LayerDataTuple:
        # get expression values
        geneid = adata.var_names.get_loc(gene)
        color_value = X[:, geneid]
        
        # create points layer
        layer = _create_points_layer(
            points=points,
            color_value=color_value,
            name=gene,
            pixel_size=pixel_size,
            size_factor=30
        )
        
        return layer
    
    @magicgui(
        call_button='Add',
        observation={'choices': obses}
        )
    def add_observations(observation=None) -> napari.types.LayerDataTuple:
        # get observation values
        color_value = adata.obs[observation].values
        # create points layer
        layer = _create_points_layer(
            points=points,
            color_value=color_value,
            name=observation,
            pixel_size=pixel_size,
            size_factor=30
        )
        
        return layer
    
    return add_genes, add_observations

# def initialize_annotation_widget(
#     ) -> FunctionGui:
#     @magicgui(
#         call_button='Add annotation'
#     )
#     def annotation_widget(
#         Label="test",
#         Class="blubb"
#     ):
#         print(Label)
#         print(Class)
    
#     return annotation_widget

@magic_factory(
        call_button='Add annotation layer',
        annot_label={'label': 'Label:'},
        class_name={'label': 'Class:'}
    )
def _annotation_widget(
    annot_label: str = "",
    class_name: str = ""
) -> napari.types.LayerDataTuple:
    # generate name
    name_pattern: str = "*{class_name} ({annot_label})"
    name = name_pattern.format(class_name=class_name, annot_label=annot_label)

    if (class_name != "") & (annot_label != ""):
        # generate shapes layer for annotation
        layer = (
            [],
            {
                'name': name,
                'shape_type': 'polygon',
                'edge_width': 10,
                'edge_color': "red",
                'face_color': 'transparent',
                'properties': {
                    'uid': np.array([], dtype='object')
                }
                }, 
            'shapes'
            )
        
        #annotation_widget.annot_label.value = ""
        _annotation_widget.class_name.value = ""
        
        return layer

    else:
        return None
