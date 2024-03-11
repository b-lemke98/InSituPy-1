from numbers import Number
from typing import List, Optional, Tuple

import dask
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

import insitupy._core.config as config

from ..image.utils import create_img_pyramid
from ..utils.palettes import CustomPalettes
from ..utils.utils import convert_to_list
from ._layers import _add_annotations_as_layer


def _create_points_layer(points, 
                         color_values: List[Number], 
                         name: str, 
                         point_names: List[str],
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
    point_names = point_names[mask]

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
            'properties': {
                "expression": color_values,
                "cell_name": point_names
                },
            'symbol': 'o',
            'size': point_size,
            'face_color': {
                "color_mode": color_mode, # workaround (see https://github.com/napari/napari/issues/6433)
                "colors": "expression"
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


def _initialize_widgets(
    xdata # xenium data object
    ) -> List[FunctionGui]:
    
    # access viewer from xeniumdata
    viewer = xdata.viewer
    
    if not hasattr(xdata, "cells"):
        add_points_widget = None
        move_to_cell_widget = None
        add_boundaries_widget = None
    else:
        # initialize data_name of viewer
        config.init_data_name()
        # initialize viewer configuration
        #config_mod.get_data_name(data_widget=select_data)
        config.update_viewer_config(xdata=xdata, 
                                    #data_name=config.current_data_name
                                    )
        
        data_names = ["main"]
        try:
            alt = xdata.alt
        except AttributeError:
            pass
        else:
            for k in alt.keys():
                data_names.append(k)
                
        @magicgui(
            call_button=False,
            data_name= {'choices': data_names, 'label': 'Dataset:'}
        )
        def select_data(
            data_name="main"
        ):
            #print(data_name)
            pass
            
        # connect key change with update function
        @select_data.data_name.changed.connect
        def update_widgets_on_data_change(event=None):
            config.current_data_name = select_data.data_name.value
            config._refresh_widgets_after_data_change(xdata, 
                                                  add_points_widget, 
                                                  add_boundaries_widget)
        
        if len(config.masks) > 0:
            @magicgui(
                call_button='Add',
                key={'choices': config.masks, 'label': 'Masks:'}
            )
            def add_boundaries_widget(
                key
            ):
                layer_name = f"{config.current_data_name}-{key}"
                
                if layer_name not in viewer.layers:
                    # get geopandas dataframe with regions
                    mask = getattr(config.boundaries, key)
                    
                    # get metadata for mask
                    metadata = config.boundaries.metadata
                    pixel_size = metadata[key]["pixel_size"]
                    
                    if not isinstance(mask, list):
                        # generate pyramid of the mask
                        mask_pyramid = create_img_pyramid(img=mask, nsubres=6)
                    else:
                        mask_pyramid = mask
                    
                    # add masks as labels to napari viewer
                    print(pixel_size, flush=True)
                    viewer.add_labels(mask_pyramid, name=layer_name, scale=(pixel_size,pixel_size))
                else:
                    print(f"Layer '{layer_name}' already in layer list.", flush=True)
        else:
            add_boundaries_widget = None
        
        @magicgui(
            call_button='Add',
            gene={'choices': config.genes, 'label': "Gene:"},
            observation={'choices': config.observations, 'label': "Observation:"},
            size={'label': 'Size [µm]'}
            )
        def add_points_widget(
            gene=None,
            observation=None,
            size=6,
            viewer=viewer
            ) -> napari.types.LayerDataTuple:
            
            # get names of cells
            cell_names = config.adata.obs_names.values
            
            layers = []
            if gene is not None:
                if gene not in viewer.layers:
                    # get expression values
                    gene_loc = config.adata.var_names.get_loc(gene)
                    color_value_gene = config.X[:, gene_loc]
                    
                    # create points layer for genes
                    gene_layer = _create_points_layer(
                        points=config.points,
                        color_values=color_value_gene,
                        name=f"{config.current_data_name}-{gene}",
                        point_names=cell_names,
                        point_size=size
                    )
                    layers.append(gene_layer)
                else:
                    print(f"Key '{gene}' already in layer list.", flush=True)
            
            if observation is not None:
                if observation not in viewer.layers:
                    # get observation values
                    color_value_obs = config.adata.obs[observation].values
                    
                    # create points layer for observations
                    obs_layer = _create_points_layer(
                        points=config.points,
                        color_values=color_value_obs,
                        name=f"{config.current_data_name}-{observation}",
                        point_names=cell_names,
                        point_size=size,
                    )
                    layers.append(obs_layer)
                else:
                    print(f"Key '{observation}' already in layer list.", flush=True)
                    
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
            if cell in config.adata.obs_names.astype(str):
                # get location of selected cell
                cell_loc = config.adata.obs_names.get_loc(cell)
                cell_position = config.points[cell_loc]
            
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
                
        def callback(event=None):
            # after the points widget is run, the widgets have to be refreshed to current data layer
            config._refresh_widgets_after_data_change(xdata,
                                                      points_widget=add_points_widget,
                                                      boundaries_widget=add_boundaries_widget
                                                      )
        if add_points_widget is not None:
            add_points_widget.call_button.clicked.connect(callback)
        if add_boundaries_widget is not None:
            add_boundaries_widget.call_button.clicked.connect(callback)
        
    
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
            tolerance: Number = 5,
            # region
            ):
            layer_name = f"region-{key}"
            
            if layer_name not in viewer.layers:
                # get geopandas dataframe with regions
                reg_df = getattr(xdata.regions, key)
                
                # simplify polygons for visualization
                reg_df["geometry"] = reg_df["geometry"].simplify(tolerance)
                
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
            else:
                print(f"Key '{key}' already in layer list.")
                
                # restore original configuration of dropdown lists
                #add_region_widget.key.value = first_region_key
                #add_region_widget.region.choices = first_regions
                
        @add_region_widget.key.changed.connect
        def update_region_on_key_change(event=None):
            _update_region_on_key_change(add_region_widget)
            
    if not hasattr(xdata, "annotations"):
        show_annotations_widget = None
    else:
        # get colorcycle for region annotations
        cmap_annotations = "Dark2"
        cmap_annot = matplotlib.colormaps[cmap_annotations]
        cc_annot = cmap_annot.colors
        
        def _update_classes_on_key_change(widget):
            current_key = widget.key.value
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
        def show_annotations_widget(key, 
                                   annot_class,
                                   tolerance: Number = 2
                                   ):
            
            # get annotation dataframe
            annot_df = getattr(xdata.annotations, key)
            
            if annot_class == "all":
                # get classes
                classes = annot_df['name'].unique()
            else:
                classes = [annot_class]
            
            # iterate through classes
            for cl in classes:
                # generate layer name
                layer_name = f"*{cl} ({key})"
                
                if layer_name not in viewer.layers:
                    # get dataframe for this class
                    class_df = annot_df[annot_df["name"] == cl]
                    
                    # simplify polygons for visualization
                    class_df["geometry"] = class_df["geometry"].simplify(tolerance)
                    
                    # add layer to viewer
                    _add_annotations_as_layer(
                        dataframe=class_df,
                        viewer=viewer,
                        layer_name=layer_name
                    )
            
        # connect key change with update function
        @show_annotations_widget.key.changed.connect
        def update_classes_on_key_change(event=None):
            _update_classes_on_key_change(show_annotations_widget)
        
        
    
    return add_points_widget, move_to_cell_widget, add_region_widget, show_annotations_widget, add_boundaries_widget, select_data #add_genes, add_observations


@magic_factory(
    call_button='Add annotation layer',
    annot_key={'label': 'Key:'},
    class_name={'label': 'Class:'}
    )
def add_new_annotations_widget(
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
        
        add_new_annotations_widget.class_name.value = ""
        
        return layer

    else:
        return None
