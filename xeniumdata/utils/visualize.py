import napari
from matplotlib.colors import rgb2hex
import numpy as np
import matplotlib
from typing import Optional, Tuple, Union, List, Dict, Any, Literal
from .utils import convert_to_list

def interactive(self,
    cell_type_key: Optional[str] = None,
    mask: Optional[List[bool]] = None,
    annotation_labels: Optional[str] = None,
    show_cells: bool = True,
    cmap_cells="tab20",
    cmap_annotations="Dark2"
    ):
    
    # subset adata
    if mask is None:
        subset = self.matrix
    else:
        subset = self.matrix[self.matrix.obs[mask]]
            
    # create viewer
    self.viewer = napari.Viewer()
        
    # add images
    for i, img_name in enumerate(self.images.metadata.keys()):
        img = getattr(self.images, img_name)
        self.viewer.add_image(
                img,
                #channel_axis=channel_axis,
                name=img_name,
                #colormap=["gray", "blue", "green"],
                rgb=self.images.metadata[img_name]["rgb"],
                contrast_limits=self.images.metadata[img_name]["contrast_limits"]
                #scale=img_scale
            )
    
    # get colorcycle for region annotations
    cmap_annot = matplotlib.colormaps[cmap_annotations]
    cc_annot = cmap_annot.colors
    
    if annotation_labels is not None:
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
    
    # get color cycle for points
    colormap = matplotlib.colormaps[cmap_cells]

    # split by high intensity and low intensity colors in tab20
    cmap1 = colormap.colors[::2]
    cmap2 = colormap.colors[1::2]

    # concatenate color cycle
    color_cycle = cmap1[:7] + cmap1[8:] + cmap2[:7] + cmap2[8:]
    
    if show_cells:
        # get point coordinates
        all_points = np.flip(subset.obsm["spatial"].copy(), axis=1) # switch x and y (napari uses [row,column])
    
        # coordinates are in µm. Convert to pixel
        #all_points /= self.metadata["pixel_size"] # "pixel_size" from experiment.xenium file
        
        if cell_type_key is not None:
            # get clusters
            clusters = list(subset.obs[cell_type_key].unique())
            colors = [rgb2hex(color_cycle[i]) for i in range(len(clusters))]
        else:
            clusters = [None]
            colors = ["gray"]
    
        for i, c in enumerate(clusters):
            if c is not None:
                # select points of this cluster
                points = all_points[subset.obs[cell_type_key] == c]
            else:
                points = all_points

            point_layer = self.viewer.add_points(points, 
                                            name=str(c),
                                            #properties=point_properties,
                                            symbol='o',
                                            size=30,
                                            face_color=colors[i],
                                            opacity=1,
                                            visible=True, 
                                            edge_width=0
                                            )
    
    napari.run()
    return self.viewer