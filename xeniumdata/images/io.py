import tifffile as tf
import sys
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any, Literal
import os
from .manipulation import resize_image

def img_resize(img,scale_factor):
    width = int(np.floor(img.shape[1] * scale_factor))
    height = int(np.floor(img.shape[0] * scale_factor))
    return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

def write_ome_tiff(
    file: Union[str, os.PathLike, Path],
    image: np.ndarray,
    axes: str = "YXS", # channels - other examples: 'TCYXS'. S for RGB channels.
    metadata: dict = {},
    subresolutions = 7, 
    subres_steps: int = 2,
    pixelsize: Optional[float] = 0.2125, # defaults to Xenium settings.
    pixelunit: Optional[str] = "µm", # usually µm
    photometric: Literal['rgb', 'minisblack', 'maxisblack'] = 'rgb', # before I had rgb here. Xenium doc says minisblack
    tile: tuple = (1024, 1024), # 1024 pixel is optimal for Xenium Explorer
    compression: Literal['jpeg', 'LZW', 'jpeg2000', None] = 'jpeg2000', # jpeg2000 is used in Xenium documentation
    overwrite: bool = False
    ):
    
    '''
    Function to write (pyramidal) OME-TIFF files.
    Code adapted from: https://github.com/cgohlke/tifffile and Xenium docs (see below).

    For parameters optimal for Xenium see: https://www.10xgenomics.com/support/software/xenium-explorer/tutorials/xe-image-file-conversion
    '''
    
    file = Path(file)
    if file.exists():
        if overwrite:
            file.unlink() # delete file
        else:
            raise FileExistsError("Input file exists already ({}).\nFor overwriting it, select `overwrite=True`".format(file))
        
    # get id of channel axis
    if not "S" in axes:
        channel_axis = axes.find("C")
    else:
        channel_axis = -1

    #fn = filename + ".ome.tif"
    with tf.TiffWriter(file, bigtiff=True) as tif:
        metadata={**metadata,
                  **{
                      'SignificantBits': 8,
                      'PhysicalSizeX': pixelsize,
                      'PhysicalSizeXUnit': pixelunit,
                      'PhysicalSizeY': pixelsize,
                      'PhysicalSizeYUnit': pixelunit,
                  }
        }
        options = dict(
            photometric=photometric,
            tile=tile,
            compression=compression,
            resolutionunit='CENTIMETER'
        )
        tif.write(
            #np.moveaxis(image, channel_axis, 0),
            image,
            subifds=subresolutions,
            resolution=(1e4 / pixelsize, 1e4 / pixelsize),
            metadata=metadata,
            **options
        )

        scale = 1
        for i in range(subresolutions):
            scale /= subres_steps
            #downsample = img_resize(image,scale),
            image = resize_image(image, scale_factor=1/subres_steps, channel_axis=channel_axis)
            tif.write(
                #np.moveaxis(downsample, channel_axis, 0),
                image,
                subfiletype=1,
                resolution=(1e4 / scale / pixelsize,1e4 / scale / pixelsize),
                **options
            )
