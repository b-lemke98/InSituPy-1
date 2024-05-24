import os
import warnings
from pathlib import Path
from typing import List, Literal, Optional, Union

import dask.array as da
import numpy as np
import tifffile as tf
import xmltodict
import zarr
from parse import *
from tifffile import TiffFile, imread

from insitupy import __version__

from .._exceptions import InvalidFileTypeError
from ..utils.utils import convert_to_list
from ..utils.utils import textformat as tf
from .utils import resize_image


def read_image(
    image
    ):
    image = Path(image)
    suffix = image.name.split(".", maxsplit=1)[-1]

    if "zarr" in suffix:
    # load image from .zarr.zip
        zipped = True if suffix == "zarr.zip" else False
        with zarr.ZipStore(image, mode="r") if zipped else zarr.DirectoryStore(image) as dirstore:
            # get components of zip store
            components = dirstore.listdir()

            if ".zarray" in components:
                # the store is an array which can be opened
                if zipped:
                    img = da.from_zarr(dirstore).persist()
                else:
                    img = da.from_zarr(dirstore)
            else:
                subres = [elem for elem in components if not elem.startswith(".")]
                img = []
                for s in subres:
                    if zipped:
                        img.append(
                            da.from_zarr(dirstore, component=s).persist()
                                    )
                    else:
                        img.append(
                            da.from_zarr(dirstore, component=s)
                                    )

            # retrieve OME metadata
            store = zarr.open(dirstore)
            meta = store.attrs.asdict()
            ome_meta = meta["OME"]
            axes = meta["axes"]

    elif suffix in ["ome.tif", "ome.tiff"]:
        # load image from .ome.tiff
        img = read_ome_tiff(path=image, levels=None)
        # read ome metadata
        with TiffFile(image) as tif:
            axes = tif.series[0].axes # get axes
            ome_meta = tif.ome_metadata # read OME metadata
            ome_meta = xmltodict.parse(ome_meta, attr_prefix="")["OME"] # convert XML to dict

    else:
        raise InvalidFileTypeError(
            allowed_types=["zarr", "zarr.zip", "ome.tif", "ome.tiff"],
            received_type=suffix
            )

    return img

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

def read_ome_tiff(
    path,
    levels: Optional[Union[List[int], int]] = None,
    new_method: bool = True
    ):
    '''
    Function to load pyramid from `ome.tiff` file.
    From: https://www.youtube.com/watch?v=8TlAAZcJnvA
    Another good resource from 10x: https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/analysis/xoa-output-understanding-outputs

    Args:
        path (str): The file path to the `ome.tiff` file.
        levels (Optional[Union[List[int], int]]): A list of integers representing the levels of the pyramid to load. If None, all levels are loaded. Default is None.
        new_method (bool): Is now the default method and uses a strategy found here: https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/analysis/xoa-output-understanding-outputs.

    Returns:
        List[dask.array.Array] or dask.array.Array: The pyramid or a single level of the pyramid, represented as Dask arrays.

    '''
    if new_method:
        pyramid = []
        l = 0
        while True:
            try:
                store = imread(path, aszarr=True, level=l, is_ome=False)
                pyramid.append(da.from_zarr(store))
                l+=1 # count up
            except IndexError:
                break

    else:
        # read store
        store = imread(path, aszarr=True)

        # Open store (root group)
        grp = zarr.open(store, mode='r')

        # Read multiscale metadata
        datasets = grp.attrs["multiscales"][0]["datasets"]

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