import os
from pathlib import Path
from typing import Union

import geopandas as gpd
import shapely

from insitupy.io.files import read_json


def read_baysor_polygons(
    file: Union[str, os.PathLike, Path]
    ) -> gpd.GeoDataFrame:

    d = read_json(file)

    # prepare output dictionary
    df = {
    "geometry": [],
    "cell": [],
    "type": [],
    "minx": [],
    "miny": [],
    "maxx": [],
    "maxy": []
    }

    for elem in d["geometries"]:
        coords = elem["coordinates"][0]

        # check if there are enough coordinates for a Polygon (some segmented cells are very small in Baysor)
        if len(coords) > 3:
            p = shapely.Polygon(coords)
            df["geometry"].append(p)
            df["type"].append("polygon")

        else:
            p = shapely.LineString(coords)
            df["geometry"].append(p)
            df["type"].append("line")
        df["cell"].append(elem["cell"])

        # extract bounding box
        bounds = p.bounds
        df["minx"].append(bounds[0])
        df["miny"].append(bounds[1])
        df["maxx"].append(bounds[2])
        df["maxy"].append(bounds[3])

    # create geopandas dataframe
    df = gpd.GeoDataFrame(df)

    return df

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

def read_proseg_polygons(
    file: Union[str, os.PathLike, Path]
    ) -> gpd.GeoDataFrame:

    d = read_json(file)

    # prepare output dictionary
    df = {
    "geometry": [],
    "cell": [],
    "type": [],
    "minx": [],
    "miny": [],
    "maxx": [],
    "maxy": []
    }
    
    for feature in d['features']:
        geometry = feature['geometry']
        properties = feature["properties"]

        if geometry['type'] == 'MultiPolygon':
            polygons = [Polygon(coords[0]) for coords in geometry['coordinates']]
            merged_geometry = unary_union(polygons).convex_hull 
            df["geometry"].append(merged_geometry)
            df["type"].append("polygon")

        elif geometry['type'] == 'Polygon':
            merged_geometry = Polygon(geometry['coordinates'][0]) 
            df["geometry"].append(merged_geometry)
            df["type"].append("polygon")
        
        df["cell"].append(properties['cell'])

        # extract bounding box
        bounds = merged_geometry.bounds
        df["minx"].append(bounds[0])
        df["miny"].append(bounds[1])
        df["maxx"].append(bounds[2])
        df["maxy"].append(bounds[3])

        
    # create geopandas dataframe
    df = gpd.GeoDataFrame(df)

    return df