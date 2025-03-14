import os
from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from anndata import AnnData

from insitupy.io.files import read_json


def _read_baysor_polygons(
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

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


def _read_proseg_polygons(
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

def _read_proseg_counts(path_counts, path_cell_metadata):
    path_counts = Path(path_counts)
    path_cell_metadata = Path(path_cell_metadata)
    # Read counts data based on file extension
    path_counts_suffix = path_counts.name.split(sep=".", maxsplit=1)[1]
    if path_counts_suffix == "parquet":
        counts = pd.read_parquet(path_counts)
    elif path_counts_suffix == "csv.gz":
        counts = pd.read_csv(path_counts, compression='gzip')
    elif path_counts_suffix == "csv":
        counts = pd.read_csv(path_counts)
    else:
        raise ValueError(f"Unexpected file ending of path_counts: {path_counts_suffix}.")

    # Read metadata based on file extension
    path_metadata_suffix = path_cell_metadata.name.split(sep=".", maxsplit=1)[1]
    if path_metadata_suffix == "parquet":
        meta = pd.read_parquet(path_cell_metadata)
    elif path_metadata_suffix == "csv.gz":
        meta = pd.read_csv(path_cell_metadata, compression='gzip')
    elif path_metadata_suffix == "csv":
        meta = pd.read_csv(path_cell_metadata)
    else:
        raise ValueError(f"Unexpected file ending of path_cell_metadata: {path_metadata_suffix}.")

    # Ensure indices are strings
    counts.index = counts.index.astype(str)
    meta.index = meta.index.astype(str)

    # Filter out unwanted columns
    counts = counts.loc[:, ~counts.columns.str.startswith('Neg')]
    counts = counts.loc[:, ~counts.columns.str.startswith('Unas')]

    # Add spatial coordinates
    obsm = {"spatial": np.stack([meta["centroid_x"].to_numpy(), meta["centroid_y"].to_numpy()], axis=1)}

    # Create AnnData object
    adata = AnnData(X=counts, obs=meta, obsm=obsm)

    # set the cell column as index
    adata.obs.set_index("cell", inplace=True)
    adata.obs_names = adata.obs_names.astype(str)
    adata.obs_names.name = None # remove the name of the index

    return adata