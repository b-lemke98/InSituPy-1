import numpy as np
import cv2
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from typing import Optional, Tuple, Union, List, Dict, Any, Literal
from PIL import Image
import dask.array as da
import numpy as np
import cv2
import numpy as np

def resize_image(img: NDArray, 
                 dim: Tuple[int, int] = None, 
                 scale_factor: float = None, 
                 channel_axis: int = -1
                 ):
    '''
    Resize image by scale_factor
    '''
    if channel_axis != -1:
        # move channel axis to last position
        img = np.moveaxis(img, channel_axis, -1)
        
    assert img.dtype in [np.dtype('uint16'), np.dtype('uint8')], \
        "Image must have one of the following numpy data types: `dtype('uint8)` or `dtype('uint16)`. \
            Otherwise cv2.resize shows an error."
    
    if isinstance(img, da.Array):
        img = img.compute() # load into memory
        
    # make sure the image is np.uint8
    #img = img.astype(np.uint8)
    
    if dim is None and scale_factor is not None:
        width = int(img.shape[1] * scale_factor)
        height = int(img.shape[0] * scale_factor)
        dim = (width, height)
    
    # do resizing
    img = cv2.resize(img, dim)
        
    if channel_axis != -1:
        # move channel axis back to original position
        img = np.moveaxis(img, -1, channel_axis)

    return img

def fit_image_to_size_limit(image: NDArray, size_limit: int, return_scale_factor: bool = True):
    # resize image if necessary (warpAffine has a size limit for the image that is transformed)
    orig_shape_image = image.shape
    xy_shape_image = orig_shape_image[:2]
    
    sf_image = (size_limit-1) / np.max(xy_shape_image)
    new_shape = [int(elem * sf_image) for elem in xy_shape_image]

    # if image has three dimensions (RGB) add third dimensions after resizing
    if len(image.shape) == 3:
            new_shape += [image.shape[-1]]
    new_shape = tuple(new_shape)

    # resize image
    resized_image = resize_image(image, dim=(new_shape[1], new_shape[0]))
    
    if return_scale_factor:
        return resized_image, sf_image
    else:
        return resized_image
    
def convert_to_8bit(img, save_mem=True, verbose=False):
    '''
    Convert numpy array image to 8bit.
    '''
    if not img.dtype == np.dtype('uint8'):
        if save_mem:
            # for a 16-bit image at least int32 is necessary for signed integers because the value range is [-65535,...,0,...,65535]
            # or uint16 can be used as unsigned integer with only positive values
            img = np.uint16(img)
        img = (img / img.max()) * 255
        img = np.uint8(img)
    else:
        if verbose:
            print("Image is already 8-bit. Not changed.", flush=True)
    return img

    