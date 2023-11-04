from typing import Optional, Tuple, Union, List, Dict, Any, Literal
from pathlib import Path
import os
from os.path import relpath
import scanpy as sc
import pandas as pd
from ..utils.utils import decode_robust_series, convert_to_list
from ..utils.annotations import read_qupath_annotation
from parse import *
from ..utils.data import ImageData, BoundariesData, AnnotationData
from pandas.api.types import is_numeric_dtype
import warnings
import shutil
from ..images.io import write_ome_tiff
import json

def read_matrix(self, 
                read_cells: bool = True
                ):
    if self.from_xeniumdata:
        self.matrix = sc.read(self.path / self.xd_metadata["matrix"])
    else:
        # extract parameters from metadata
        pixel_size = self.metadata["pixel_size"]
        cf_zarr_path = self.path / self.metadata["xenium_explorer_files"]["cell_features_zarr_filepath"]
        cf_h5_path = cf_zarr_path.parent / cf_zarr_path.name.replace(".zarr.zip", ".h5")

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            # read matrix data
            self.matrix = sc.read_10x_h5(cf_h5_path)
        
        if read_cells:
            # read cell information
            cells_zarr_path = self.path / self.metadata["xenium_explorer_files"]["cells_zarr_filepath"]
            cells_parquet_path = cells_zarr_path.parent / cells_zarr_path.name.replace(".zarr.zip", ".parquet")
            cells = pd.read_parquet(cells_parquet_path)
            
            # transform cell ids from bytes to str
            cells = cells.set_index("cell_id")

            # make sure that the indices are decoded strings
            if is_numeric_dtype(cells.index):
                cells.index = cells.index.astype(str)
            else:
                cells.index = decode_robust_series(cells.index)
            
            # add information to anndata observations
            self.matrix.obs = pd.merge(left=self.matrix.obs, right=cells, left_index=True, right_index=True)
        
        # transfer coordinates to .obsm
        coord_cols = ["x_centroid", "y_centroid"]
        self.matrix.obsm["spatial"] = self.matrix.obs[coord_cols].values
        self.matrix.obsm["spatial"] /= pixel_size
        self.matrix.obs.drop(coord_cols, axis=1, inplace=True)
    
def read_images(self,
                names: Union[Literal["all", "nuclei"], str] = "all", # here a specific image can be chosen
                dapi_type: str = "focus"
                ):
    if self.from_xeniumdata:
        img_files = list(self.xd_metadata["images"].values())
        img_names = list(self.xd_metadata["images"].keys())
    else:
        if names == "nuclei":
            img_keys = [f"morphology_{dapi_type}_filepath"]
            img_names = ["nuclei"]
        else:
            # get available keys for registered images in metadata
            img_keys = [elem for elem in self.metadata["images"] if elem.startswith("registered")]
            
            # extract image names from keys and add nuclei
            img_names = ["nuclei"] + [elem.split("_")[1] for elem in img_keys]
            
            # add dapi image key
            img_keys = [f"morphology_{dapi_type}_filepath"] + img_keys
            
            if names != "all":
                # make sure keys is a list
                names = convert_to_list(names)
                # select the specified keys
                mask = [elem in names for elem in img_names]
                img_keys = [elem for m, elem in zip(mask, img_keys) if m]
                img_names = [elem for m, elem in zip(mask, img_names) if m]
                
        # get path of image files
        img_files = [self.metadata["images"][k] for k in img_keys]
        
    # load image into ImageData object
    self.images = ImageData(self.path, img_files, img_names)
    
    
def read_transcripts(self,
                     transcript_filename: str = "transcripts.parquet"
                     ):
    if self.from_xeniumdata:
        self.transcripts = pd.read_parquet(self.path / self.xd_metadata["transcripts"])
    else:
        # read transcripts
        self.transcripts = pd.read_parquet(self.path / transcript_filename)
        
        # decode columns
        self.transcripts = self.transcripts.apply(lambda x: decode_robust_series(x), axis=0)
        
        # convert coordinates into pixel coordinates
        coord_cols = ["x_location", "y_location", "z_location"]
        self.transcripts[coord_cols] = self.transcripts[coord_cols].apply(lambda x: x / self.metadata["pixel_size"])
        
def read_boundaries(self):
    # read boundaries data
    self.boundaries = BoundariesData(path=self.path, pixel_size=self.metadata["pixel_size"])
    
def read_annotations(self,
                     annot_path: Union[str, os.PathLike, Path] = "../annotations",
                     suffix: str = ".geojson",
                     pattern_annotation_file: str = "annotation-{slide_id}__{region_id}__{name}"
                     ):
    annot_path = Path(annot_path)
    
    # check if the annotation path exists. If it does not, first assume that it is a relative path and check that.
    if not annot_path.is_dir():
        annot_path = Path(os.path.normpath(os.path.join(self.path, annot_path)))
        assert annot_path.is_dir(), "`annot_path` is neither a direct path nor a relative path."
    
    annot_path = annot_path
    self.annotations = AnnotationData()
    for file in annot_path.glob(f"*{suffix}"):
        if self.slide_id in str(file.stem) and (self.region_id in str(file.stem)):
            parsed = parse(pattern_annotation_file, file.stem)
            annot_name = parsed.named["name"]
            
            # read annotation and store in dictionary
            self.annotations.add_annotation(read_qupath_annotation(file=file), annot_name)
            
def read_all(self, verbose: bool = True):
    read_funcs = [elem for elem in dir(self) if elem.startswith("read_")]
    read_funcs = [elem for elem in read_funcs if elem != "read_all"]
    
    # check if there is an annotations folder
    if len(list(self.path.parent.glob("annotations"))) == 0:
        read_funcs.remove("read_annotations")
        print("No folder named `annotations` found. Function `read_annotations()` was skipped.", flush=True)
        
    for f in read_funcs:
        if verbose: 
            print(f"Running {f}()", flush=True)
        func = getattr(self, f)
        func()
        
 
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
    metadata["region_id"] = self.region_id
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
            annot_file = annot_path / f"{n}.parquet"
            annot_df.to_parquet(annot_file)
            metadata["annotations"][n] = Path(relpath(annot_file, path)).as_posix()
            
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
        