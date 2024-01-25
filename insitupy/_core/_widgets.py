from typing import Tuple, List, Optional

import matplotlib
import napari
import numpy as np
import pandas as pd
from anndata import AnnData
from magicgui import magic_factory, magicgui
from magicgui.widgets import FunctionGui
from matplotlib.colors import rgb2hex
from napari.types import LayerDataTuple
from pandas.api.types import is_numeric_dtype
from scipy.sparse import issparse
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import LinearRing, Polygon

from ..utils.palettes import CustomPalettes
from ..utils.utils import convert_to_list


def _create_points_layer(points, 
                         color_values, 
                         name, 
                         point_size: int = 6, # is in scale unit (so mostly µm)
                         opacity: float = 1,
                         visible: bool = True,
                         edge_width: float = 0,
                         edge_color: str = 'red'
                         ) -> LayerDataTuple:
    
    # remove entries with NaN
    mask = pd.notnull(color_values)
    color_values = color_values[mask]
    points = points[mask]

    # check if the data should be plotted categorical or continous
    if is_numeric_dtype(color_values):
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
        climits = [0, np.percentile(color_values, 95)]
    
    # generate point layer
    layer = (
        points, 
        {
            'name': name,
            'properties': {"color_value": color_values},
            'symbol': 'o',
            'size': point_size,
            'face_color': {
                "color_mode": color_mode, # workaround (see https://github.com/napari/napari/issues/6433)
                "colors": "color_value"
                },
            'face_color_cycle': color_cycle,
            'face_colormap': color_map,
            'face_contrast_limits': climits,
            'opacity': opacity,
            'visible': visible,
            'edge_width': edge_width,
            'edge_color': edge_color
            }, 
        'points'
        )
    return layer

def _initialize_point_widgets(
    xdata # xenium data object
    ) -> List[FunctionGui]:
    
    # access viewer from xeniumdata
    viewer = xdata.viewer
    
    if not hasattr(xdata, "cells"):
        add_points_widget = None
        move_to_cell_widget = None
    else:
        # access adata and viewer from xeniumdata
        adata = xdata.cells.matrix
        
        # get point coordinates
        points = np.flip(adata.obsm["spatial"].copy(), axis=1) # switch x and y (napari uses [row,column])
    
        # get expression matrix
        if issparse(adata.X):
            X = adata.X.toarray()
        else:
            X = adata.X
        
        @magicgui(
            call_button='Add',
            gene={'choices': adata.var_names.tolist(), 'label': "Gene:"},
            observation={'choices': adata.obs.columns.tolist(), 'label': "Observation:"},
            size={'label': 'Size [µm]'}
            )
        def add_points_widget(
            gene=None,
            observation=None,
            size=6,
            viewer=viewer
            ) -> napari.types.LayerDataTuple:
            

            
            layers = []
            if gene is not None:
                if gene not in viewer.layers:
                    # get expression values
                    gene_loc = adata.var_names.get_loc(gene)
                    color_value_gene = X[:, gene_loc]
                
                    # create points layer for genes
                    gene_layer = _create_points_layer(
                        points=points,
                        color_values=color_value_gene,
                        name=gene,
                        point_size=size
                    )
                    layers.append(gene_layer)
                else:
                    print(f"Key '{gene}' already in layer list.")
            
            if (observation is not None):
                if observation not in viewer.layers:
                    # get observation values
                    color_value_obs = adata.obs[observation].values                
                    
                    # create points layer for observations
                    obs_layer = _create_points_layer(
                        points=points,
                        color_values=color_value_obs,
                        name=observation,
                        point_size=size,
                    )
                    layers.append(obs_layer)
                else:
                    print(f"Key '{observation}' already in layer list.")
                        
            return layers
        
        @magicgui(
            call_button='Show',
            cell={'label': "Cells:"},
            zoom={'label': 'Zoom:'},
            highlight={'label': 'Highlight'}
            )
        def move_to_cell_widget(
            cell="",
            zoom=5,
            highlight=True,
            #viewer=viewer
        ) -> Optional[napari.types.LayerDataTuple]:
            if cell in adata.obs_names.astype(str):
                # get location of selected cell
                cell_loc = adata.obs_names.get_loc(cell)
                cell_position = points[cell_loc]
            
                # move center of camera to cell position
                viewer.camera.center = (0, cell_position[0], cell_position[1])
                viewer.camera.zoom = zoom
                
                if highlight:
                    name = f"cell-{cell}"
                    if name not in viewer.layers:
                        viewer.add_points(
                            data=np.array([cell_position]),
                            name=name,
                            size=6,
                            face_color=[0,0,0,0],
                            opacity=1,
                            edge_color='red',
                            edge_width=0.1
                        )
            else:
                print(f"Cell '{cell}' not found in `xeniumdata.cells.matrix.obs_names()`.")
    
    if not hasattr(xdata, "regions"):
        add_region_widget = None
    else:
        # get colormap for regions
        cmap_regions = matplotlib.colormaps["tab10"]
        
        def _update_region_on_key_change(widget):
            current_key = widget.key.value
            widget.region.choices = sorted(xdata.regions.metadata[current_key]['classes'])
            pass

        # extract region keys
        region_keys = list(xdata.regions.metadata.keys())
        first_region_key = list(region_keys)[0] # for dropdown menu
        #first_regions = xdata.regions.metadata[first_region_key]['classes']
        
        @magicgui(
            call_button='Show',
            key={"choices": region_keys, "label": "Key:"},
            #region={"choices": first_regions, "label": "Regions:"}
        )
        def add_region_widget(
            key, 
            # region
            ):
            layer_name = f"region-{key}"
            
            if layer_name not in viewer.layers:
                # get geopandas dataframe with regions
                reg_df = getattr(xdata.regions, key)
                
                # iterate through shapes and collect them as list
                shapes_list = []
                uids_list = []
                names_list = []
                for uid, row in reg_df.iterrows():
                    # get coordinates
                    polygon = row["geometry"]
                    
                    if isinstance(polygon, MultiPolygon):
                        raise TypeError("Region is a shapely 'MultiPolygon'. This is not supported in regions. Make sure to only have simple polygons as regions.")
                    
                    # extract exterior coordinates from shapely object
                    # Note: the last coordinate is removed since it is identical with the first
                    # in shapely objects, leading sometimes to visualization bugs in napari
                    exterior_array = np.array([polygon.exterior.coords.xy[1].tolist()[:-1],
                                            polygon.exterior.coords.xy[0].tolist()[:-1]]).T
                    
                    # collect data
                    shapes_list.append(exterior_array)
                    uids_list.append(uid)
                    names_list.append(row["name"])
                    
                # determine hexcolor for this region key
                hexcolor = rgb2hex([elem / 255 for elem in cmap_regions(region_keys.index(key))])
                hexcolor = rgb2hex(cmap_regions(region_keys.index(key)))
                
                text = {
                    'string': '{name}',
                    'anchor': 'upper_left',
                    #'translation': [-5, 0],
                    'size': 8,
                    'color': hexcolor,
                    }
                
                viewer.add_shapes(
                    shapes_list,
                    name=layer_name,
                    properties={
                        'uid': uids_list,
                        'name': names_list
                    },
                    shape_type="polygon",
                    edge_width=10,
                    edge_color=hexcolor,
                    face_color='transparent',
                    text=text
                    )
                
                # restore original configuration of dropdown lists
                #add_region_widget.key.value = first_region_key
                #add_region_widget.region.choices = first_regions
                
        @add_region_widget.key.changed.connect
        def update_region_on_key_change(event=None):
            _update_region_on_key_change(add_region_widget)
            
    if not hasattr(xdata, "annotations"):
        add_annotations_widget = None
    else:
        # get colorcycle for region annotations
        cmap_annotations = "Dark2"
        cmap_annot = matplotlib.colormaps[cmap_annotations]
        cc_annot = cmap_annot.colors
        
        def _update_classes_on_key_change(widget):
            current_key = widget.key.value
            print(current_key)
            widget.annot_class.choices = ["all"] + sorted(xdata.annotations.metadata[current_key]['classes'])
        
        # extract region keys
        annot_keys = list(xdata.annotations.metadata.keys())
        first_annot_key = list(annot_keys)[0] # for dropdown menu
        first_classes = ["all"] + sorted(xdata.annotations.metadata[first_annot_key]['classes'])
        
        @magicgui(
            call_button='Show',
            key={"choices": annot_keys, "label": "Key:"},
            annot_class={"choices": first_classes, "label": "Class:"}
        )
        def add_annotations_widget(key, annot_class):
            
            # get annotation dataframe
            annot_df = getattr(xdata.annotations, key)
            
            if annot_class == "all":
                # get classes
                classes = annot_df['name'].unique()
            else:
                classes = [annot_class]
            
            # iterate through classes
            for cl in classes:
                
                layer_name = f"*{cl} ({key})"
                
                if layer_name not in viewer.layers:
                    # get dataframe for this class
                    class_df = annot_df[annot_df["name"] == cl]
                    
                    # iterate through annotations of this class and collect them as list
                    shape_list = []
                    color_list = []
                    uid_list = []
                    type_list = [] # list to store whether the polygon is exterior or interior
                    for uid, row in class_df.iterrows():
                        # get coordinates
                        polygon = row["geometry"]
                        #uid = row["id"]
                        hexcolor = rgb2hex([elem / 255 for elem in row["color"]])
                        
                        # check if polygon is a MultiPolygon or just a simple Polygon object
                        if isinstance(polygon, MultiPolygon):
                            poly_list = list(polygon.geoms)
                        elif isinstance(polygon, Polygon):
                            poly_list = [polygon]
                        else:
                            raise ValueError(f"Input must be a Polygon or MultiPolygon object. Received: {type(polygon)}")
                        
                        for p in poly_list:
                            # extract exterior coordinates from shapely object
                            # Note: the last coordinate is removed since it is identical with the first
                            # in shapely objects, leading sometimes to visualization bugs in napari
                            exterior_array = np.array([p.exterior.coords.xy[1].tolist()[:-1],
                                                    p.exterior.coords.xy[0].tolist()[:-1]]).T
                            #exterior_array *= pixel_size # convert to length unit
                            shape_list.append(exterior_array)  # collect shape
                            color_list.append(hexcolor)  # collect corresponding color
                            uid_list.append(uid)  # collect corresponding unique id
                            type_list.append("exterior")
                            
                            # if polygon has interiors, plot them as well
                            # for information on donut-shaped polygons in napari see: https://forum.image.sc/t/is-it-possible-to-generate-doughnut-shapes-in-napari-shapes-layer/88834
                            if len(p.interiors) > 0:
                                for linear_ring in p.interiors:
                                    if isinstance(linear_ring, LinearRing):
                                        interior_array = np.array([linear_ring.coords.xy[1].tolist()[:-1],
                                                                linear_ring.coords.xy[0].tolist()[:-1]]).T
                                        #interior_array *= pixel_size # convert to length unit
                                        shape_list.append(interior_array)  # collect shape
                                        color_list.append(hexcolor)  # collect corresponding color
                                        uid_list.append(uid)  # collect corresponding unique id
                                        type_list.append("interior")
                                    else:
                                        ValueError(f"Input must be a LinearRing object. Received: {type(linear_ring)}")
                        
                    viewer.add_shapes(shape_list, 
                                    name=layer_name,
                                    properties={
                                        'uid': uid_list, # list with uids
                                        'type': type_list # list giving information on whether the polygon is interior or exterior
                                    },
                                    shape_type='polygon', 
                                    edge_width=10,
                                    edge_color=color_list,
                                    face_color='transparent'
                                    )
            
        # connect key change with update function
        @add_annotations_widget.key.changed.connect
        def update_classes_on_key_change(event=None):
            _update_classes_on_key_change(add_annotations_widget)
        
        
    
    return add_points_widget, move_to_cell_widget, add_region_widget, add_annotations_widget #add_genes, add_observations


@magic_factory(
    call_button='Add annotation layer',
    annot_key={'label': 'Key:'},
    class_name={'label': 'Class:'}
    )
def _annotation_widget(
    annot_key: str = "",
    class_name: str = ""
) -> napari.types.LayerDataTuple:
    # generate name
    name_pattern: str = "*{class_name} ({annot_key})"
    name = name_pattern.format(class_name=class_name, annot_key=annot_key)

    if (class_name != "") & (annot_key != ""):
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
        
        _annotation_widget.class_name.value = ""
        
        return layer

    else:
        return None
