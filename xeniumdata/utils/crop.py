import napari

def crop(self, 
         shape_layer: str = "crop"
         ):
    '''
    Function to crop the XeniumData object.
    '''
    # extract shape layer for cropping from napari viewer
    crop_shape = self.viewer.layers[shape_layer]
    
    # check the structure of the shape object
    assert len(crop_shape.data) == 1, "More than one region was selected. Abort."
    crop_window = crop_shape.data[0]
    assert isinstance(crop_shape, napari.layers.Shapes), "Selected layer is not a shape layer."
    
    # extract x and y limits from the shape (assuming a rectangle)
    xlim = (crop_window[:, 1].min(), crop_window[:, 1].max())
    ylim = (crop_window[:, 0].min(), crop_window[:, 0].max())
    
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