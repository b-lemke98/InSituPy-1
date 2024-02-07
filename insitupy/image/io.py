import os
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import tifffile as tf
import warnings

from .utils import resize_image


def write_ome_tiff(
    file: Union[str, os.PathLike, Path],
    image: np.ndarray,
    axes: str = "YXS", # channels - other examples: 'TCYXS'. S for RGB channels. 'YX' for grayscale image.
    metadata: dict = {},
    subresolutions = 7, 
    subres_steps: int = 2,
    pixelsize: Optional[float] = 1, # defaults to Xenium settings.
    pixelunit: Optional[str] = None, # usually µm
    #significant_bits: Optional[int] = 16,
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
    # check dtype of image
    if image.dtype not in [np.dtype('uint16'), np.dtype('uint8')]:
        warnings.warn("Image does not have dtype 'uint8' or 'uint16'. Is converted to 'uint16'.")
        
        if image.dtype == np.dtype('int8'):
            image = image.astype('uint8')
        else:
            image = image.astype('uint16')
        
    # determine significant bits variable - is important that Xenium explorer correctly distinguishes between 8 bit and 16 bit
    if image.dtype == np.dtype('uint8'):
        significant_bits = 8
    else:
        significant_bits = 16
    
    file = Path(file)
    if file.exists():
        if overwrite:
            file.unlink() # delete file
        else:
            raise FileExistsError("Output file exists already ({}).\nFor overwriting it, select `overwrite=True`".format(file))
        
    # create metadata
    if pixelsize != 1:        
        metadata = {
            **metadata,
            **{
                'PhysicalSizeX': pixelsize,
                'PhysicalSizeY': pixelsize
            }
        }
    if pixelunit is not None:
        metadata = {
            **metadata,
            **{
                'PhysicalSizeXUnit': pixelunit,
                'PhysicalSizeYUnit': pixelunit
            }
        }
    if (significant_bits is not None) & ("SignificantBits" not in metadata.keys()):
        metadata = {
            **metadata,
            **{
                'SignificantBits': significant_bits
            }
        }
    

    with tf.TiffWriter(file, bigtiff=True) as tif:
        options = dict(
            photometric=photometric,
            tile=tile,
            compression=compression,
            resolutionunit='CENTIMETER',
        )
        tif.write(
            image,
            subifds=subresolutions,
            resolution=(1e4 / pixelsize, 1e4 / pixelsize),
            metadata=metadata,
            **options
        )

        scale = 1
        for i in range(subresolutions):
            scale /= subres_steps
            image = resize_image(image, scale_factor=1/subres_steps, axes=axes)
            tif.write(
                image,
                subfiletype=1,
                resolution=(1e4 / scale / pixelsize,1e4 / scale / pixelsize),
                **options
            )
