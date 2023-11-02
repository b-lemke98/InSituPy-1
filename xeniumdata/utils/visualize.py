import napari
from napari.utils.notifications import Notification
from matplotlib.colors import rgb2hex
import numpy as np
import matplotlib
from typing import Optional, Tuple, Union, List, Dict, Any, Literal
from .utils import convert_to_list
from .exceptions import XeniumDataMissingObject
from ..palettes import CustomPalettes
from scipy.sparse import issparse
from magicgui import magicgui

def show(self,
    keys: Optional[str] = None,
    annotation_labels: Optional[str] = None,
    show_images: bool = True,
    show_cells: bool = False,
    scalebar: bool = True,
    pixel_size: float = None, # if none, extract from metadata
    unit: str = "µm",
    cmap_annotations="Dark2"
    ):
    
    # get information about pixel size
    if (pixel_size is None) & (scalebar):
        # extract pixel_size
        pixel_size = float(self.metadata["pixel_size"])
    else:
        pixel_size = 1
        
    # get available genes
    genes = self.matrix.var_names.tolist()
    obses = self.matrix.obs.columns.tolist()
    
    # get point coordinates
    points = np.flip(self.matrix.obsm["spatial"].copy(), axis=1) * pixel_size # switch x and y (napari uses [row,column])
    
    # get expression matrix
    if issparse(self.matrix.X):
        X = self.matrix.X.toarray()
    else:
        X = self.matrix.X
            
    # create viewer
    self.viewer = napari.Viewer()
    
    @magicgui(
        call_button='Add',
        gene={'choices': genes},
        )
    def add_genes(gene=None) -> napari.types.LayerDataTuple:
        # get expression values
        geneid = self.matrix.var_names.get_loc(gene)
        expr = X[:, geneid]
        
        # set color settings for continuous data
        color_map = "viridis"
        color_cycle = None
        climits = [0, np.percentile(expr, 95)]
        
        # generate point layer
        layer = (
            points, 
            {
                'name': gene,
                'properties': {"expr": expr},
                'symbol': 'o',
                'size': 30 * pixel_size,
                'face_color': "expr",
                'face_color_cycle': color_cycle,
                'face_colormap': color_map,
                'face_contrast_limits': climits,
                'opacity': 1,
                'visible': True,
                'edge_width': 0
                }, 
            'points'
            )
        return layer
    
    @magicgui(
        call_button='Add',
        observation={'choices': obses}
        )
    def add_observations(observation=None) -> napari.types.LayerDataTuple:
        # get observation values
        expr = self.matrix.obs[observation].values
        
        # get color cycle for categorical data
        palettes = CustomPalettes()
        color_cycle = getattr(palettes, "tab20_mod").colors
        color_map = None
        climits = None
        
        # generate point layer
        layer = (
            points, 
            {
                'name': observation,
                'properties': {"expr": expr},
                'symbol': 'o',
                'size': 30 * pixel_size,
                'face_color': "expr",
                'face_color_cycle': color_cycle,
                'face_colormap': color_map,
                'face_contrast_limits': climits,
                'opacity': 1,
                'visible': True,
                'edge_width': 0
                }, 
            'points'
            )
        return layer
                
    if show_images:
        # add images
        if not hasattr(self, "images"):
            raise XeniumDataMissingObject("images")
            
        image_keys = self.images.metadata.keys()
        for i, img_name in enumerate(image_keys):
            img = getattr(self.images, img_name)
            visible = False if i < len(image_keys) - 1 else True # only last image is set visible
            self.viewer.add_image(
                    img,
                    #channel_axis=channel_axis,
                    name=img_name,
                    #colormap=["gray", "blue", "green"],
                    rgb=self.images.metadata[img_name]["rgb"],
                    contrast_limits=self.images.metadata[img_name]["contrast_limits"],
                    scale=(pixel_size, pixel_size),
                    visible=visible
                )
    
    # optionally: add cells as points
    if show_cells or keys is not None:
        if not hasattr(self, "matrix"):
            raise XeniumDataMissingObject("matrix")       

        # convert keys to list
        keys = convert_to_list(keys)
        
        # get point coordinates
        points = np.flip(self.matrix.obsm["spatial"].copy(), axis=1) * pixel_size # switch x and y (napari uses [row,column])
        
        # get expression matrix
        if issparse(self.matrix.X):
            X = self.matrix.X.toarray()
        else:
            X = self.matrix.X

        point_layers = {}
        for i, k in enumerate(keys):
            visible = False if i < len(keys) - 1 else True # only last image is set visible
            # get expression values
            if k in self.matrix.obs.columns:
                expr = self.matrix.obs[k].values
                
                # get color cycle for categorical data
                palettes = CustomPalettes()
                color_cycle = getattr(palettes, "tab20_mod").colors
                color_map = None
                climits = None
            else:
                geneid = self.matrix.var_names.get_loc(k)
                expr = X[:, geneid]
                color_map = "viridis"
                color_cycle = None
                climits = [0, np.percentile(expr, 95)]
            
            point_properties = {
                "expr": expr,
                #"confidence": subset.X.toarray()[:, 1]
            }

            point_layers[k] = self.viewer.add_points(points,
                                            name=k,
                                            properties=point_properties,
                                            symbol='o',
                                            size=30 * pixel_size,
                                            face_color="expr",
                                            face_color_cycle=color_cycle,
                                            face_colormap=color_map,
                                            face_contrast_limits=climits,
                                            opacity=1,
                                            visible=visible,
                                            edge_width=0
                                            )

    
    if annotation_labels is not None:
        # get colorcycle for region annotations
        cmap_annot = matplotlib.colormaps[cmap_annotations]
        cc_annot = cmap_annot.colors
        
        if annotation_labels == "all":
            annotation_labels = self.annotations.labels
        annotation_labels = convert_to_list(annotation_labels)
        for annotation_label in annotation_labels:
            annot_df = getattr(self.annotations, annotation_label)
            # add annotations as shapes
            for i, row in annot_df.iterrows():
                # get metadata
                annot_name = row["name"]
                shape = row["geometry"]
                hexcolor = rgb2hex([elem / 255 for elem in row["color"]])
                
                # extract coordinates from shapely object
                shape_array = np.array([shape.exterior.coords.xy[1].tolist(), shape.exterior.coords.xy[0].tolist()]).T
                #shape_arrays = [np.array([c.exterior.coords.xy[1].tolist(), c.exterior.coords.xy[0].tolist()]).T for c in shape]
        
                self.viewer.add_shapes(shape_array, 
                                name=annot_name, 
                                shape_type='polygon', 
                                edge_width=50,
                                edge_color=hexcolor,
                                face_color='transparent'
                                )
    
    if scalebar:
        # add scale bar
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = unit
        
    # set maximum height of widget to prevent the widget from having a large distance
    add_genes.max_height = 100
    add_observations.max_height = 100
    
    # add widgets to napari window
    self.viewer.window.add_dock_widget(add_genes, name="Add genes", area="right")
    self.viewer.window.add_dock_widget(add_observations, name="Add observations", area="right")
    
    napari.run()
    return self.viewer