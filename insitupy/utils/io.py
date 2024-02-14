import json
import os
import shutil
from pathlib import Path
from typing import List, Union

import dask.array as da
import matplotlib.pyplot as plt
import pandas as pd
import zarr

from .utils import nested_dict_numpy_to_list


#TODO: `load_pyramid` should be moved to .image.io
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

def write_dict_to_json(
    dictionary: dict,
    file: Union[str, os.PathLike, Path],
    ):
    try:
        dict_json = json.dumps(dictionary, indent=4)
        with open(file, "w") as metafile:
                metafile.write(dict_json)
    except TypeError:
        # one reason for this type error could be that there are ndarrays in the dict
        # convert them to lists
        nested_dict_numpy_to_list(dictionary)
        
        dict_json = json.dumps(dictionary, indent=4)
        with open(file, "w") as metafile:
                metafile.write(dict_json)
        
def check_overwrite_and_remove_if_true(
    path: Union[str, os.PathLike, Path], 
    overwrite: bool = False
    ):
    path = Path(path)
    if path.exists():
        if overwrite:
            if path.is_dir():
                shutil.rmtree(path) # delete directory
            elif path.is_file():
                path.unlink() # delete file
            else:
                raise ValueError(f"Path is neither a directory nor a file. What is it? {str(path)}")
        else:
            raise FileExistsError(f"The output file already exists at {path}. To overwrite it, please set the `overwrite` parameter to True."
)
    

def save_and_show_figure(savepath, fig, save_only=False, show=True, dpi_save=300, save_background=None, tight=True):
    #if fig is not None and axis is not None:
    #    return fig, axis
    #elif savepath is not None:
    if tight:
        fig.tight_layout()

    if savepath is not None:
        print("Saving figure to file " + savepath)

        # create path if it does not exist
        Path(os.path.dirname(savepath)).mkdir(parents=True, exist_ok=True)

        # save figure
        plt.savefig(savepath, dpi=dpi_save,
                    facecolor=save_background, bbox_inches='tight')
        print("Saved.")
    if save_only:
        plt.close(fig)
    elif show:
        return plt.show()
    else:
        return