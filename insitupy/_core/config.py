import dask
import numpy as np
from scipy.sparse import issparse

from insitupy import WITH_NAPARI

if WITH_NAPARI:
    import napari

def init_data_name():
    global current_data_name
    current_data_name = "main"

# set viewer configurations
def update_viewer_config(
    xdata
    ):
    if current_data_name == "main":
        # access adata, viewer and metadata from xeniumdata
        global adata
        adata = xdata.cells.matrix
        global boundaries
        boundaries = xdata.cells.boundaries
    else:
        adata = xdata.alt[current_data_name].matrix
        boundaries = xdata.alt[current_data_name].boundaries

    # get genes and observations
    global genes, observations, value_dict
    genes = adata.var_names.tolist()
    observations = adata.obs.columns.tolist()
    value_dict = {
        "genes": genes,
        "obs": observations
    }

    # get point coordinates
    global points
    points = np.flip(adata.obsm["spatial"].copy(), axis=1) # switch x and y (napari uses [row,column])

    # get expression matrix
    global X
    if issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X

    global masks
    masks = []
    for n in boundaries.metadata.keys():
        b = getattr(boundaries, n)
        if isinstance(b, dask.array.core.Array) or np.all([isinstance(elem, dask.array.core.Array) for elem in b]):
            masks.append(n)

def _refresh_widgets_after_data_change(xdata, points_widget, boundaries_widget):
    update_viewer_config(xdata)

    # set choices
    points_widget.gene.choices = genes
    points_widget.observation.choices = observations
    boundaries_widget.key.choices = masks

    # reset the currently selected key to None
    points_widget.gene.value = None
    points_widget.observation.value = None

    # set only the last layer visible
    point_layers = [elem for elem in xdata.viewer.layers if isinstance(elem, napari.layers.points.points.Points)]
    n_point_layers = len(point_layers)

    for i, l in enumerate(point_layers):
        if i < n_point_layers-1:
            l.visible = False