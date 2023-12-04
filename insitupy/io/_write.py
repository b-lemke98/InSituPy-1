from typing import Optional, Tuple, Union, List, Dict, Any, Literal
from pathlib import Path
import os
from os.path import relpath
from parse import *
import shutil
from ..images.io import write_ome_tiff
from ..io.io import write_qupath_geojson
import json
import insitupy

def save(self,
         path: Union[str, os.PathLike, Path],
         overwrite: bool = False,
         zip: bool = False
         ):
    '''
    Function to save the XeniumData object.
    
    Args:
        path: Path to save the data to.
    '''
    # check if the path already exists    
    path = Path(path)
    if path.exists():
        if overwrite:
            shutil.rmtree(path) # delete directory
            if zip:
                zippath = path.with_suffix(".zip")
                if zippath.exists():
                    zippath.unlink() # remove zip file
        else:
            raise FileExistsError("Output file exists already ({}).\nFor overwriting it, select `overwrite=True`".format(path))
    
    # create output directory if it does not exist yet
    path.mkdir(parents=True, exist_ok=True)
    
    # create a metadata dictionary
    metadata = {}
    
    # store basic information about experiment
    metadata["slide_id"] = self.slide_id
    metadata["sample_id"] = self.sample_id
    metadata["path"] = str(self.path)
    
    # save images
    if hasattr(self, "images"):
        img_path = (path / "images")
        img_path.mkdir(parents=True, exist_ok=True) # create image directory
        
        metadata["images"] = {}
        for n, img_metadata in self.images.metadata.items():
            # extract image
            img = getattr(self.images, n)[0]
            
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
            write_ome_tiff(img_path / filename, img, photometric=photometric, axes=axes, metadata=selected_metadata, overwrite=overwrite)
            
            # collect metadata
            metadata["images"][n] = Path(relpath(img_path / filename, path)).as_posix()
            
    # save matrix
    if hasattr(self, "matrix"):
        mtx_path = (path / "matrix")
        mtx_path.mkdir(parents=True, exist_ok=True) # create directory
        mtx_file = mtx_path / "matrix.h5ad"
        self.matrix.write(mtx_file)
        metadata["matrix"] = Path(relpath(mtx_file, path)).as_posix()
        
    # save transcripts
    if hasattr(self, "transcripts"):
        trans_path = (path / "transcripts")
        trans_path.mkdir(parents=True, exist_ok=True) # create directory
        trans_file = trans_path / "transcripts.parquet"
        self.transcripts.to_parquet(trans_file)
        metadata["transcripts"] = Path(relpath(trans_file, path)).as_posix()
        
    # save boundaries
    if hasattr(self, "boundaries"):
        bound_path = (path / "boundaries")
        bound_path.mkdir(parents=True, exist_ok=True) # create directory
        
        metadata["boundaries"] = {}
        for n in ["cells", "nuclei"]:
            bound_df = getattr(self.boundaries, n)
            bound_file = bound_path / f"{n}.parquet"
            bound_df.to_parquet(bound_file)
            metadata["boundaries"][n] = Path(relpath(bound_file, path)).as_posix()
            
    # save annotations
    if hasattr(self, "annotations"):
        annot_path = (path / "annotations")
        annot_path.mkdir(parents=True, exist_ok=True) # create directory
        
        metadata["annotations"] = {}
        for n in self.annotations.labels:
            annot_df = getattr(self.annotations, n)
            # annot_file = annot_path / f"{n}.parquet"
            # annot_df.to_parquet(annot_file)
            annot_file = annot_path / f"{n}.geojson"
            write_qupath_geojson(dataframe=annot_df, file=annot_file)
            metadata["annotations"][n] = Path(relpath(annot_file, path)).as_posix()
            
    # save version of InSituPy
    metadata["version"] = insitupy.__version__
            
    # Optionally: zip the resulting directory
    if zip:
        shutil.make_archive(path, 'zip', path, verbose=False)
        
    # write Xeniumdata metadata to json file
    metadata_path = path / "xeniumdata.json"
    metadata_json = json.dumps(metadata, indent=4)
    with open(metadata_path, "w") as metafile:
        metafile.write(metadata_json)
        
    # write Xenium metadata to json file
    metadata_path = path / "xenium.json"
    metadata_json = json.dumps(self.metadata, indent=4)
    with open(metadata_path, "w") as metafile:
        metafile.write(metadata_json)
        