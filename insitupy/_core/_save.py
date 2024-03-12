import shutil
from os.path import relpath
from pathlib import Path

import zarr
from parse import *

from insitupy import __version__

from ..image.io import write_ome_tiff
from ..utils.geo import write_qupath_geojson
from ..utils.io import write_dict_to_json
from ._checks import check_zip


def _save_images(imagedata, 
                 path, 
                 metadata,
                 images_as_zarr
                 ):
    img_path = (path / "images")
    #img_path.mkdir(parents=True, exist_ok=True) # create image directory
            
    savepaths = imagedata.save(path=img_path, images_as_zarr=images_as_zarr, return_savepaths=True)

    if metadata is not None:
        metadata["data"]["images"] = {}
        for n in imagedata.metadata.keys():
            s = savepaths[n]
            # collect metadata
            metadata["data"]["images"][n] = Path(relpath(s, path)).as_posix()
        
def _save_cells(cells, path, metadata):
    # create path for cells
    cells_path = path / "cells"
    
    # save cells to path and write info to metadata
    cells.save(cells_path)
    
    if metadata is not None:
        metadata["data"]["cells"] = Path(relpath(cells_path, path)).as_posix()
        
def _save_alt(attr, path, metadata):
    # create path for cells
    alt_path = path / "alt"
    
    for k, celldata in attr.items():
        cells_path = alt_path / k
        # save cells to path and write info to metadata
        celldata.save(cells_path)
    
        if metadata is not None:
            if "alt" not in metadata:
                metadata["data"]["alt"] = {}
            
            metadata["data"]["alt"][k] = Path(relpath(cells_path, path)).as_posix()
            
def _save_transcripts(transcripts, path, metadata):
    # create file path
    trans_path = (path / "transcripts")
    trans_path.mkdir(parents=True, exist_ok=True) # create directory
    trans_file = trans_path / "transcripts.parquet"
    
    # save transcripts as parquet and modify metadata
    transcripts.to_parquet(trans_file)
    
    if metadata is not None:
        metadata["data"]["transcripts"] = Path(relpath(trans_file, path)).as_posix()
    
def _save_annotations(annotations, path, metadata):
    annot_path = (path / "annotations")
    
    # save annotations
    annotations.save(annot_path)
        
    if metadata is not None:
        metadata["data"]["annotations"] = Path(relpath(annot_path, path)).as_posix()
    
def _save_regions(regions, path, metadata):
    annot_path = (path / "regions")
    
    # save annotations
    regions.save(annot_path)
        
    if metadata is not None:
        metadata["data"]["regions"] = Path(relpath(annot_path, path)).as_posix()
