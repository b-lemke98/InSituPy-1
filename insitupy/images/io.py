import os
import warnings
from pathlib import Path
from typing import List, Literal, Optional, Union

import dask.array as da
import numpy as np
import tifffile as tf
import zarr
from tifffile import imread

from ..utils.utils import convert_to_list
from .utils import resize_image


def write_ome_tiff(
    file: Union[str, os.PathLike, Path],
    image: Union[np.ndarray, da.core.Array, List[da.core.Array]],
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
    # check if the image is an image pyramid
    if isinstance(image, list):
        # if it is a pyramid, select only the highest resolution image
        image = image[0]
        
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

def read_ome_tiff(path, 
                 levels: Optional[Union[List[int], int]] = None
                 ):
    '''    
    Function to load pyramid from `ome.tiff` file.
    From: https://www.youtube.com/watch?v=8TlAAZcJnvA
    
    Args:
        path (str): The file path to the `ome.tiff` file.
        levels (Optional[Union[List[int], int]]): A list of integers representing the levels of the pyramid to load. If None, all levels are loaded. Default is None.

    Returns:
        List[dask.array.Array] or dask.array.Array: The pyramid or a single level of the pyramid, represented as Dask arrays.

    '''
    # read store
    store = imread(path, aszarr=True)
    
    # Open store (root group)
    grp = zarr.open(store, mode='r')

    # Read multiscale metadata
    datasets = grp.attrs["multiscales"][0]["datasets"]
    
    # pyramid = [
    #     da.from_zarr(store, component=d["path"])
    #     for d in datasets
    # ]
    
    if levels is None:
        levels = range(0, len(datasets))
    # make sure level is a list
    levels = convert_to_list(levels)
    
    # extract images as pyramid list
    pyramid = [
        da.from_zarr(store, component=datasets[l]["path"])
        for l in levels
    ]
    
    # if pyramid has only one element, return only this image
    if len(pyramid) == 1:
        pyramid = pyramid[0]

    return pyramid