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
from .format import ImageAxes

def resize_image(img: NDArray, 
                 dim: Tuple[int, int] = None, 
                 scale_factor: float = None, 
                 axes = "YXS"
                 ):
    '''
    Resize image by scale_factor
    '''
    # read and interpret the image axes pattern
    image_axes = ImageAxes(pattern=axes)
    channel_axis = image_axes.C
    
    if (channel_axis is not None) & (channel_axis != len(img.shape)-1):
        # move channel axis to last position if it is not there already
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
        
    if (channel_axis is not None) & (channel_axis != -1):
        # move channel axis back to original position
        img = np.moveaxis(img, -1, channel_axis)

    return img

def fit_image_to_size_limit(image: NDArray, 
                            axes: str,  # description of axes, e.g. YXS for RGB, CYX for IF, TYXS for time-series RGB
                            size_limit: int, 
                            return_scale_factor: bool = True
                            ):
    '''
    Function to resize image if necessary (warpAffine has a size limit for the image that is transformed).
    '''
    # get information about channels
    #image_axes = ImageAxes(pattern=axes)
    
    orig_shape_image = image.shape
    xy_shape_image = orig_shape_image[:2]
    
    sf_image = (size_limit-1) / np.max(xy_shape_image)
    new_shape = [int(elem * sf_image) for elem in xy_shape_image]

    # if image has three dimensions (RGB) add third dimensions after resizing
    if len(image.shape) == 3:
            new_shape += [image.shape[-1]]
    new_shape = tuple(new_shape)

    # resize image
    resized_image = resize_image(image, dim=(new_shape[1], new_shape[0]), axes=axes)
    
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

def scale_to_max_width(image: np.ndarray, 
                       axes: str,  # description of axes, e.g. YXS for RGB, CYX for IF, TYXS for time-series RGB
                       max_width: int = 4000,
                       use_square_area: bool = False,
                       #channel_axis: int = 2,
                       verbose: bool = True,
                       print_spacer: str = ""
                       ):
    '''
    Function to scale image to a maximum width or square area.
    '''
    image_axes = ImageAxes(pattern=axes)
    image_yx = (image.shape[image_axes.Y], image.shape[image_axes.X])
    # if image_axes.C is not None:
    #     image_xy = tuple([image.shape[i] for i in range(3) if i != 0])  # extract image shape based on channel axis
    # else:
    #     # if the channel_axis is None, the image does not have a channel axis, meaning it is a grayscale image
    #     image_xy = image.shape
        
    #image_xy = image.shape[:2] # extract image shape assuming that the channels are in third dimension
    #num_dim = len(image.shape)
    
    # if num_dim == 3:
    #     assert image.shape[-1] == 3, "Image has three dimensions but the third channel is not 3. No RGB?"
    
    if not use_square_area:
        # scale to the longest side of the image. Not good for very elongated images.
        if np.max(image_yx) > max_width:
            new_shape = tuple([int(elem / np.max(image_yx) * max_width) for elem in image_yx])
        else:
            new_shape = image.shape
            
    else:
        # use the square area of the maximum width as measure for rescaling. Better for elongated images.
        max_square_area = max_width ** 2
        
        # calculate new dimensions based on the maximum square area
        long_idx = np.argmax(image_yx)  # search for position of longest dimension
        short_idx = np.argmin(image_yx)  # same for shortest
        long_side = image_yx[long_idx]  # extract longest side
        short_side = image_yx[short_idx]  # extract shortest
        dim_ratio = short_side / long_side  # calculate ratio between the two sides.
        new_long_side = int(np.sqrt(max_square_area / dim_ratio))  # calculate the length of the new longer side based on area
        new_short_side = int(new_long_side * dim_ratio) # calculate length of new shorter side based on the longer one
        
        # create new shape
        new_shape = [None, None]
        new_shape[long_idx] = new_long_side
        new_shape[short_idx] = new_short_side
        new_shape = tuple(new_shape)
                
    # resizing - caution: order of dimensions is reversed in OpenCV compared to numpy
    image_scaled = resize_image(img=image, dim=(new_shape[1], new_shape[0]), axes=axes)
    print(f"{print_spacer}Rescaled to following dimensions: {image_scaled.shape}") if verbose else None
    
    return image_scaled
