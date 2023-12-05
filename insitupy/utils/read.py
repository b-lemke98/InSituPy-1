import json
import os
from pathlib import Path
from typing import Union

import dask.array as da
import zarr


def load_pyramid(store):
    '''
    Function to load pyramid.
    From: https://www.youtube.com/watch?v=8TlAAZcJnvA
    '''
    # Open store (root group)
    grp = zarr.open(store, mode='r')

    # Read multiscale metadata
    datasets = grp.attrs["multiscales"][0]["datasets"]

    return [
        da.from_zarr(store, component=d["path"])
        for d in datasets
    ]
    
def read_json(
    file: Union[str, os.PathLike, Path],
    ) -> dict:
    '''
    Function to load json files as dictionary.
    '''
    # load metadata file
    with open(file, "r") as metafile:
        metadata = json.load(metafile)
        
    return metadata