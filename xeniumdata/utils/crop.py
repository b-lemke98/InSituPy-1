import napari
import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any, Literal
from .exceptions import XeniumDataRepeatedCropError, WrongNapariLayerTypeError

def crop(self, 
         shape_layer: Optional[str] = None,
         xlim: Optional[Tuple[int, int]] = None,
         ylim: Optional[Tuple[int, int]] = None
         ):
    '''
    Function to crop the XeniumData object.
    '''
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
        crop_shape = self.viewer.layers[shape_layer]
        
        # check the structure of the shape object
        assert len(crop_shape.data) == 1, "More than one region was selected. Abort."
        crop_window = crop_shape.data[0]
        if not isinstance(crop_shape.dtype(), napari.layers.Shapes):
            raise WrongNapariLayerTypeError(found=crop_shape, wanted=napari.layers.Shapes)
        
        # extract x and y limits from the shape (assuming a rectangle)
        xlim = (crop_window[:, 1].min(), crop_window[:, 1].max())
        ylim = (crop_window[:, 0].min(), crop_window[:, 0].max())
        
    # if the object was previously cropped, check if the current window is identical with the previous one
    ###>> to be done
    if np.all([elem in self.metadata.keys() for elem in ["cropping_xlim", "cropping_ylim"]]):
        # test whether the limits are identical
        if (xlim == self.metadata["cropping_xlim"]) & (ylim == self.metadata["cropping_ylim"]):
            raise XeniumDataRepeatedCropError(xlim, ylim)
    
    # infer mask from cell coordinates
    cell_coords = self.matrix.obsm['spatial'].copy()
    xmask = (cell_coords[:, 0] >= xlim[0]) & (cell_coords[:, 0] <= xlim[1])
    ymask = (cell_coords[:, 1] >= ylim[0]) & (cell_coords[:, 1] <= ylim[1])
    mask = xmask & ymask
    
    # select 
    self.matrix = self.matrix[mask, :].copy()
    
    # move origin again to 0 by subtracting the lower limits from the coordinates
    cell_coords = self.matrix.obsm['spatial'].copy()
    cell_coords[:, 0] -= xlim[0]
    cell_coords[:, 1] -= ylim[0]
    self.matrix.obsm['spatial'] = cell_coords
    
    # synchronize other data modalities to match the anndata matrix
    if hasattr(self, "boundaries"):
        self.boundaries.sync_to_matrix(cell_ids=self.matrix.obs_names, xlim=xlim, ylim=ylim)
        
    if hasattr(self, "transcripts"):
        # infer mask for selection
        xmask = (self.transcripts["x_location"] >= xlim[0]) & (self.transcripts["x_location"] <= xlim[1])
        ymask = (self.transcripts["y_location"] >= ylim[0]) & (self.transcripts["y_location"] <= ylim[1])
        mask = xmask & ymask
        
        # select
        self.transcripts = self.transcripts.loc[mask, :].copy()
        
        # move origin again to 0 by subtracting the lower limits from the coordinates
        self.transcripts["x_location"] -= xlim[0]
        self.transcripts["y_location"] -= ylim[0]
        
    if hasattr(self, "images"):
        self.images.crop(xlim=xlim, ylim=ylim)
    
    # add information about cropping to metadata
    self.metadata["cropping_xlim"] = xlim
    self.metadata["cropping_ylim"] = ylim