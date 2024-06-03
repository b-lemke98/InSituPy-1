from numbers import Number
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.colors import rgb2hex
from napari.types import LayerDataTuple
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import LinearRing, Polygon

from insitupy import WITH_NAPARI
from insitupy.utils.palettes import data_to_rgba

if WITH_NAPARI:
    import napari

    def _add_annotations_as_layer(
        dataframe: pd.DataFrame,
        viewer: napari.Viewer,
        layer_name: str
        ):
        # iterate through annotations of this class and collect them as list
        shape_list = []
        color_list = []
        uid_list = []
        type_list = [] # list to store whether the polygon is exterior or interior
        for uid, row in dataframe.iterrows():
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


    def _create_points_layer(points,
                            color_values: List[Number],
                            name: str,
                            point_names: List[str],
                            point_size: int = 6, # is in scale unit (so mostly µm)
                            opacity: float = 1,
                            visible: bool = True,
                            edge_width: float = 0,
                            edge_color: str = 'red',
                            upper_climit_pct: int = 99,
                            cmap: str = "viridis"
                            ) -> LayerDataTuple:

        # remove entries with NaN
        # mask = pd.notnull(color_values)
        # color_values = color_values[mask]
        # points = points[mask]
        # point_names = point_names[mask]

        # get colors
        colors = data_to_rgba(data=color_values)

        # generate point layer
        layer = (
            points,
            {
                'name': name,
                'properties': {
                    "value": color_values,
                    "cell_name": point_names
                    },
                'symbol': 'o',
                'size': point_size,
                'face_color': colors,
                #     {
                #     "color_mode": color_mode, # workaround (see https://github.com/napari/napari/issues/6433)
                #     "colors": "value"
                #     },
                # 'face_color_cycle': color_cycle,
                # 'face_colormap': color_map,
                # 'face_contrast_limits': climits,
                'opacity': opacity,
                'visible': visible,
                'edge_width': edge_width,
                'edge_color': edge_color
                },
            'points'
            )
        return layer

    def _update_points_layer(
        layer: napari.layers.Layer,
        new_color_values: List[Number],
        new_name: Optional[str] = None,
        upper_climit_pct: int = 99,
        # cmap: str = "viridis"
        ) -> None:
        # get the RGBA colors for the new values
        new_colors = data_to_rgba(data=new_color_values, upper_climit_pct=upper_climit_pct)

        # change the colors of the layer
        layer.face_color = new_colors

        # change properties of layer
        new_props = layer.properties.copy()
        new_props['value'] = new_color_values
        layer.properties = new_props

        if new_name is not None:
            layer.name = new_name