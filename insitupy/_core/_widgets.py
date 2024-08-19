from numbers import Number
from typing import List, Optional
from warnings import warn

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import rgb2hex
from shapely.geometry.multipolygon import MultiPolygon

import insitupy._core.config as config
from insitupy import WITH_NAPARI
from insitupy._core._layers import _create_points_layer, _update_points_layer

from ..images.utils import create_img_pyramid

if WITH_NAPARI:
    import napari
    from magicgui import magic_factory, magicgui
    from magicgui.widgets import FunctionGui

    from ._layers import _add_annotations_as_layer

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
            config.set_viewer_config(xdata=xdata,
                                        #data_name=config.current_data_name
                                        )
            config.init_recent_selections()

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
                        viewer.add_labels(mask_pyramid, name=layer_name, scale=(pixel_size,pixel_size))
                    else:
                        print(f"Layer '{layer_name}' already in layer list.", flush=True)
            else:
                add_boundaries_widget = None

            def _update_values_on_key_change(widget):
                current_key = widget.key.value
                widget.value.choices = config.value_dict[current_key]

            @magicgui(
                call_button='Add',
                key={'choices': ["genes", "obs", "obsm"], 'label': 'Key:'},
                value={'choices': config.genes, 'label': "Value:"},
                size={'label': 'Size [µm]'},
                recent={'choices': [""], 'label': "Recent:"},
                )
            def add_points_widget(
                key="genes",
                value=None,
                size=6,
                recent=None,
                viewer=viewer
                ) -> napari.types.LayerDataTuple:

                # get names of cells
                cell_names = config.adata.obs_names.values

                #layers_to_add = []
                if value is not None or recent is not None:
                    if value is None:
                        key = recent.split(":", maxsplit=1)[0]
                        value = recent.split(":", maxsplit=1)[1]
                    #if gene not in viewer.layers:
                    # get expression values
                    if key == "genes":
                        gene_loc = config.adata.var_names.get_loc(value)
                        color_value = config.X[:, gene_loc]
                    elif key == "obs":
                        color_value = config.adata.obs[value]
                    elif key == "obsm":
                        #TODO: Implement it for obsm
                        obsm_key = value.split("-", maxsplit=1)[0]
                        obsm_col = value.split("-", maxsplit=1)[1]
                        data = config.adata.obsm[obsm_key]

                        if isinstance(data, pd.DataFrame):
                            color_value = data[obsm_col]
                        elif isinstance(data, np.ndarray):
                            color_value = data[:, int(obsm_col)-1]
                        else:
                            warn("Data in `obsm` needs to be either pandas DataFrame or numpy array to be parsed.")
                        pass
                    else:
                        print("Unknown key selected.", flush=True)

                    new_layer_name = f"{config.current_data_name}-{value}"

                    # get layer names from the current data
                    layer_names_for_current_data = [elem.name for elem in viewer.layers if elem.name.startswith(config.current_data_name)]

                    # select only point layers
                    layer_names_for_current_data = [elem for elem in layer_names_for_current_data if isinstance(viewer.layers[elem], napari.layers.points.points.Points)]

                    # save last addition to add it to recent in the callback
                    config.recent_selections.append(f"{key}:{value}")

                    if len(layer_names_for_current_data) == 0:

                        # create points layer for genes
                        gene_layer = _create_points_layer(
                            points=config.points,
                            color_values=color_value,
                            #name=f"{config.current_data_name}-{gene}",
                            name=new_layer_name,
                            point_names=cell_names,
                            point_size=size,
                            upper_climit_pct=99
                        )
                        return gene_layer
                        #layers_to_add.append(gene_layer)
                    else:
                        #print(f"Key '{gene}' already in layer list.", flush=True)
                        # update the points layer
                        layer = viewer.layers[layer_names_for_current_data[0]]
                        _update_points_layer(
                            layer=layer,
                            new_color_values=color_value,
                            new_name=new_layer_name,
                        )

            @add_points_widget.key.changed.connect
            def update_values_on_key_change(event=None):
                _update_values_on_key_change(add_points_widget)

            @magicgui(
                call_button='Show',
                cell={'label': "Cell:"},
                zoom={'label': 'Zoom:'},
                highlight={'label': 'Highlight'}
                )
            def move_to_cell_widget(
                cell="",
                zoom=5,
                highlight=True,
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
            show_regions_widget = None
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
            def show_regions_widget(
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

            @show_regions_widget.key.changed.connect
            def update_region_on_key_change(event=None):
                _update_region_on_key_change(show_regions_widget)

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
                        class_df = annot_df.loc[annot_df["name"] == cl]

                        # simplify polygons for visualization
                        class_df["geometry"] = class_df["geometry"].simplify(tolerance)

                        # extract scale
                        scale = class_df.iloc[0]["scale"]

                        # add layer to viewer
                        _add_annotations_as_layer(
                            dataframe=class_df,
                            viewer=viewer,
                            layer_name=layer_name,
                            scale=scale
                        )

            # connect key change with update function
            @show_annotations_widget.key.changed.connect
            def update_classes_on_key_change(event=None):
                _update_classes_on_key_change(show_annotations_widget)



        return add_points_widget, move_to_cell_widget, show_regions_widget, show_annotations_widget, add_boundaries_widget, select_data #add_genes, add_observations


    @magic_factory(
        call_button='Add annotation layer',
        key={"choices": ["Shapes", "Points"], "label": "Type:"},
        annot_key={'label': 'Key:'},
        class_name={'label': 'Class:'}
        )
    def add_new_annotations_widget(
        key: str = "Shapes",
        annot_key: str = "TestKey",
        class_name: str = "TestClass",
    ) -> napari.types.LayerDataTuple:
        # generate name
        name_pattern: str = "*{class_name} ({annot_key})"
        name = name_pattern.format(class_name=class_name, annot_key=annot_key)

        if (class_name != "") & (annot_key != ""):
            if key == "Shapes":
                # generate shapes layer for annotation
                layer = (
                    [],
                    {
                        'name': name,
                        'shape_type': 'polygon',
                        'edge_width': 40,
                        'edge_color': 'red',
                        'face_color': 'transparent',
                        'scale': (config.pixel_size, config.pixel_size),
                        'properties': {
                            'uid': np.array([], dtype='object')
                        }
                        },
                    'shapes'
                    )
            elif key == "Points":
                # generate points layer for annotation
                layer = (
                    [],
                    {
                        'name': name,
                        'size': 100,
                        'edge_color': 'blue',
                        'face_color': 'blue',
                        'scale': (config.pixel_size, config.pixel_size),
                        'properties': {
                            'uid': np.array([], dtype='object')
                        }
                        },
                    'points'
                    )

            add_new_annotations_widget.class_name.value = ""

            return layer

        else:
            return None
