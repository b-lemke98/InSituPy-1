import numpy as np
import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any, Literal
from skimage.color import rgb2hed, hed2rgb
from .manipulation import convert_to_8bit

def deconvolve_he(
    img: np.ndarray,
    return_type: Literal["grayscale", "greyscale", "rgb"] = "grayscale",
    convert: bool = True # convert to 8-bit
    ) -> np.ndarray:
    '''
    Deconvolves H&E image to separately extract hematoxylin and eosin stainings.
    
    from: https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_ihc_color_separation.html
    
    --------------
    Returns:
    For return_type "grayscale": Numpy array with shape (h, w, 3) where the 3 channels correspond to 
        hematoxylin, eosin and a third channel
        
    For return_type "rgb": Three separate RGB images as numpy array. Order: Hematoxylin, eosin, and third green channel.
    '''
    # perform deconvolution
    ihc_hed = rgb2hed(img)
    
    if return_type in ["grayscale", "greyscale"]:
        # extract hematoxylin channel and convert to 8-bit
        ihc_h = ihc_hed[:, :, 0] # hematoxylin
        ihc_e = ihc_hed[:, :, 1] # eosin
        ihc_d = ihc_hed[:, :, 2] # DAB
    
    elif return_type == "rgb":
        # Create an RGB image for each of the stains
        null = np.zeros_like(ihc_hed[:, :, 0])
        ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
        ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
        ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))
        
    else:
        raise ValueError('Unknown `return_type`. Possible values: "grayscale" or "rgb".')
        
    if convert:
        ihc_h = convert_to_8bit(ihc_h, save_mem=False)
        ihc_e = convert_to_8bit(ihc_e, save_mem=False)
        ihc_d = convert_to_8bit(ihc_d, save_mem=False)

    return ihc_h, ihc_e, ihc_d