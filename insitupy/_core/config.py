import dask
import numpy as np
from scipy.sparse import issparse


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
    global genes, observations
    genes = adata.var_names.tolist()
    observations = adata.obs.columns.tolist()
    
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
    points_widget.gene.choices = genes
    points_widget.observation.choices = observations
    boundaries_widget.key.choices = masks
    