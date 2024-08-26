import geopandas as gpd
import numpy as np
from shapely.affinity import scale as scale_func

from insitupy import InSituData


def calc_distance_of_cells_from(
    data: InSituData,
    annotation_key: str,
    annotation_class: str,
    key_to_save: str = None
    ):

    # create geopandas points from cells
    x = data.cells.matrix.obsm["spatial"][:, 0]
    y = data.cells.matrix.obsm["spatial"][:, 1]
    cells = gpd.points_from_xy(x, y)

    # retrieve annotation information
    annot_df = data.annotations.get(annotation_key)
    class_df = annot_df[annot_df["name"] == annotation_class]

    # calculate distance of cells to their closest point
    scaled_geometries = [
        scale_func(geometry, xfact=scale[0], yfact=scale[1], origin=(0,0))
        for geometry, scale in zip(class_df["geometry"], class_df["scale"])
        ]
    dists = np.array([cells.distance(geometry) for geometry in scaled_geometries])
    min_dists = dists.min(axis=0)

    # add results to CellData
    if key_to_save is None:
        key_to_save = f"dist_from_{annotation_class}"
    data.cells.matrix.obs[key_to_save] = min_dists
    print(f'Save distances to `.cells.matrix.obs["{key_to_save}"]`')