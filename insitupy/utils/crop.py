import napari
import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any, Literal
from .exceptions import XeniumDataRepeatedCropError, WrongNapariLayerTypeError, NotOneElementError
from shapely import Polygon, affinity

def crop(self, 
         shape_layer: Optional[str] = None,
         xlim: Optional[Tuple[int, int]] = None,
         ylim: Optional[Tuple[int, int]] = None,
         inplace: bool = False
         ):
    '''
    Function to crop the XeniumData object.
    '''
    # check if the changes are supposed to be made in place or not
    with_viewer = False
    if inplace:
        _self = self
    else:
        if hasattr(self, "viewer"):
            with_viewer = True
            viewer_copy = self.viewer.copy() # copy viewer to transfer it to new object for cropping
        _self = self.copy()
        if with_viewer:
            _self.viewer = viewer_copy
            print(viewer_copy)
        
    # assert that either shape_layer is given or xlim/ylim
    assert np.any([elem is not None for elem in [shape_layer, xlim, ylim]]), "No values given for either `shape_layer` or `xlim/ylim`."
    
    if shape_layer is not None:
        use_shape = True
    else:
        # if xlim or ylim is not none, assert that both are not None
        if xlim is not None or ylim is not None:
            assert np.all([elem is not None for elem in [xlim, ylim]])
            use_shape = False
    
    if use_shape:
        # extract shape layer for cropping from napari viewer
        crop_shape = _self.viewer.layers[shape_layer]
        
        # check the structure of the shape object
        if len(crop_shape.data) != 1:
            raise NotOneElementError(crop_shape.data)
        
        # select the shape from list
        crop_window = crop_shape.data[0]
        
        # check the type of the element
        if not isinstance(crop_shape, napari.layers.Shapes):
            raise WrongNapariLayerTypeError(found=type(crop_shape), wanted=napari.layers.Shapes)
        
        # extract x and y limits from the shape (assuming a rectangle)
        xlim = (crop_window[:, 1].min(), crop_window[:, 1].max())
        ylim = (crop_window[:, 0].min(), crop_window[:, 0].max())
        
    # if the object was previously cropped, check if the current window is identical with the previous one
    if np.all([elem in _self.metadata.keys() for elem in ["cropping_xlim", "cropping_ylim"]]):
        # test whether the limits are identical
        if (xlim == _self.metadata["cropping_xlim"]) & (ylim == _self.metadata["cropping_ylim"]):
            raise XeniumDataRepeatedCropError(xlim, ylim)
    
    if hasattr(_self, "matrix"):
        # infer mask from cell coordinates
        cell_coords = _self.matrix.obsm['spatial'].copy()
        xmask = (cell_coords[:, 0] >= xlim[0]) & (cell_coords[:, 0] <= xlim[1])
        ymask = (cell_coords[:, 1] >= ylim[0]) & (cell_coords[:, 1] <= ylim[1])
        mask = xmask & ymask
        
        # select 
        _self.matrix = _self.matrix[mask, :].copy()
        
        # move origin again to 0 by subtracting the lower limits from the coordinates
        cell_coords = _self.matrix.obsm['spatial'].copy()
        cell_coords[:, 0] -= xlim[0]
        cell_coords[:, 1] -= ylim[0]
        _self.matrix.obsm['spatial'] = cell_coords
    
    # synchronize other data modalities to match the anndata matrix
    if hasattr(_self, "boundaries"):
        _self.boundaries.sync_to_matrix(cell_ids=_self.matrix.obs_names, xlim=xlim, ylim=ylim)
        
    if hasattr(_self, "transcripts"):
        # infer mask for selection
        xmask = (_self.transcripts["x_location"] >= xlim[0]) & (_self.transcripts["x_location"] <= xlim[1])
        ymask = (_self.transcripts["y_location"] >= ylim[0]) & (_self.transcripts["y_location"] <= ylim[1])
        mask = xmask & ymask
        
        # select
        _self.transcripts = _self.transcripts.loc[mask, :].copy()
        
        # move origin again to 0 by subtracting the lower limits from the coordinates
        _self.transcripts["x_location"] -= xlim[0]
        _self.transcripts["y_location"] -= ylim[0]
        
    if hasattr(_self, "images"):
        _self.images.crop(xlim=xlim, ylim=ylim)
    

    
    if hasattr(_self, "annotations"):
        limit_poly = Polygon([(xlim[0], ylim[0]), (xlim[1], ylim[0]), (xlim[1], ylim[1]), (xlim[0], ylim[1])])
        
        for i, n in enumerate(_self.annotations.labels):
            annotdf = getattr(_self.annotations, n)
            
            # select annotations that intersect with the selected area
            mask = [limit_poly.intersects(elem) for elem in annotdf["geometry"]]
            annotdf = annotdf.loc[mask, :].copy()
            
            # move origin to zero after cropping
            annotdf["geometry"] = annotdf["geometry"].apply(affinity.translate, xoff=-xlim[0], yoff=-ylim[0])
            
            # add new dataframe back to annotations object
            setattr(_self.annotations, n, annotdf)
            
            # update metadata
            _self.annotations.labels[i] = n
            _self.annotations.n_annotations[i] = len(annotdf)
            _self.annotations.classes[i] = annotdf.name.unique()
            
            
    # add information about cropping to metadata
    _self.metadata["cropping_xlim"] = xlim
    _self.metadata["cropping_ylim"] = ylim
            
    
    if not inplace:
        if hasattr(self, "viewer"):
            del _self.viewer # delete viewer
        return _self    
            