import napari
from matplotlib.colors import rgb2hex
import numpy as np
import matplotlib
from typing import Optional, Tuple, Union, List, Dict, Any, Literal

def interactive(self, 
                  groupby: str = "cell_type", 
                  mask: Optional[List[bool]] = None, 
                  #show_images: List[bool] = [True, True, True], 
                  show_shapes: bool = True, 
                  show_points: bool = True,
                  #imgds: None,
                  annot_dict=None,
                  cmap_points="tab20",
                  cmap_regions="Dark2"
                  ):
    
    # subset adata
    if mask is None:
        subset = self.matrix
    else:
        subset = self.matrix[self.matrix.obs[mask]]

    # get color cycle for points
    colormap = matplotlib.colormaps[cmap_points]

    # split by high intensity and low intensity colors in tab20
    cmap1 = colormap.colors[::2]
    cmap2 = colormap.colors[1::2]

    # concatenate color cycle
    color_cycle = cmap1[:7] + cmap1[8:] + cmap2[:7] + cmap2[8:]

    # get colorcycle for region annotations
    cmap_annot = matplotlib.colormaps[cmap_regions]
    cc_annot = cmap_annot.colors
    
    # create viewer
    viewer = napari.Viewer()
    
    # add images
    for i, img_name in enumerate(self.images.metadata.keys()):
        img = getattr(self.images, img_name)
        viewer.add_image(
                img, 
                #channel_axis=channel_axis,
                name=img_name,
                #colormap=["gray", "blue", "green"], 
                rgb=self.images.metadata[img_name]["rgb"],
                contrast_limits=self.images.metadata[img_name]["contrast_limits"]
                #scale=img_scale
            )
    
    # if show_shapes:
    #     # add annotations as shapes
    #     for i, (annot_name, shape) in enumerate(annot_dict.items()):
    #         # extract coordinates from shapely object
    #         shape_arrays = [np.array([c.exterior.coords.xy[1].tolist(), c.exterior.coords.xy[0].tolist()]).T for c in shape]
    
    #         viewer.add_shapes(shape_arrays, 
    #                           name=annot_name, 
    #                           shape_type='polygon', 
    #                           edge_width=50,
    #                           edge_color=rgb2hex(cc_annot[i]),
    #                           face_color='transparent'
    #                          )
    
    if show_points:
        # get point coordinates
        all_points = np.flip(subset.obsm["spatial"].copy(), axis=1) # switch x and y
    
        # coordinates are in µm. Convert to pixel
        #all_points /= self.metadata["pixel_size"] # "pixel_size" from experiment.xenium file
    
        # get clusters
        #clusters = list(subset.obs[groupby].unique().categories)
    
        #for i, c in enumerate(clusters):
        # select points of this cluster
        #points = all_points[subset.obs[groupby] == c]
        points = all_points

        point_layer = viewer.add_points(points, 
                                        #name=str(c),
                                        #properties=point_properties,
                                        symbol='o',
                                        size=30,
                                        #size=res,
                                        #face_color=rgb2hex(color_cycle[i]),
                                        face_color='r',
                                        #face_colormap="tab10",
                                        opacity=1,
                                        visible=True, 
                                        edge_width=0
                                        )
    
    napari.run()