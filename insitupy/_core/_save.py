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
    img_path.mkdir(parents=True, exist_ok=True) # create image directory
    
    if metadata is not None:
        metadata["images"] = {}
    for n, img_metadata in imagedata.metadata.items():
        # extract image
        img = getattr(imagedata, n)
        if isinstance(img, list):
            img = img[0]
            
        if images_as_zarr:
            filename = Path(img_metadata["file"]).name.split(".")[0] + ".zarr.zip"
            
            with zarr.ZipStore(img_path / filename, mode="w") as zipstore:
                # save image data in zipstore
                img.to_zarr(zipstore)
                
                # open zarr store save metadata in zarr store
                store = zarr.open(zipstore, mode="a")
                for k,v in img_metadata.items():
                    store.attrs[k] = v
                
        else:
            # get file name for saving
            filename = Path(img_metadata["file"]).name
            # retrieve image metadata for saving
            photometric = 'rgb' if img_metadata['rgb'] else 'minisblack'
            axes = img_metadata['axes']
            
            # retrieve OME metadata
            ome_meta_to_retrieve = ["SignificantBits", "PhysicalSizeX", "PhysicalSizeY", "PhysicalSizeXUnit", "PhysicalSizeYUnit"]
            pixel_meta = img_metadata["OME"]["Image"]["Pixels"]
            selected_metadata = {key: pixel_meta[key] for key in ome_meta_to_retrieve if key in pixel_meta}
            
            # write images as OME-TIFF
            write_ome_tiff(img_path / filename, img, 
                        photometric=photometric, axes=axes, 
                        metadata=selected_metadata, overwrite=False)
            
        if metadata is not None:
            # collect metadata
            metadata["images"][n] = Path(relpath(img_path / filename, path)).as_posix()
        
def _save_cells(cells, path, metadata):
    # create path for cells
    cells_path = path / "cells"
    
    # save cells to path and write info to metadata
    cells.save(cells_path)
    
    if metadata is not None:
        metadata["cells"] = Path(relpath(cells_path, path)).as_posix()
        
def _save_alt(attr, path, metadata):
    # create path for cells
    alt_path = path / "alt"
    
    for k, celldata in attr.items():
        cells_path = alt_path / k
        # save cells to path and write info to metadata
        celldata.save(cells_path)
    
        if metadata is not None:
            if "alt" not in metadata:
                metadata["alt"] = {}
            
            metadata["alt"][k] = Path(relpath(cells_path, path)).as_posix()
            
def _save_transcripts(transcripts, path, metadata):
    # create file path
    trans_path = (path / "transcripts")
    trans_path.mkdir(parents=True, exist_ok=True) # create directory
    trans_file = trans_path / "transcripts.parquet"
    
    # save transcripts as parquet and modify metadata
    transcripts.to_parquet(trans_file)
    
    if metadata is not None:
        metadata["transcripts"] = Path(relpath(trans_file, path)).as_posix()
    
def _save_annotations(annotations, path, metadata):
    annot_path = (path / "annotations")
    
    # save annotations
    annotations.save(annot_path)
        
    if metadata is not None:
        metadata["annotations"] = Path(relpath(annot_path, path)).as_posix()
    
def _save_regions(regions, path, metadata):
    annot_path = (path / "regions")
    
    # save annotations
    regions.save(annot_path)
        
    if metadata is not None:
        metadata["regions"] = Path(relpath(annot_path, path)).as_posix()
