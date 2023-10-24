from typing import Optional, Tuple, Union, List, Dict, Any, Literal
from pathlib import Path
import os
import scanpy as sc
import pandas as pd
from .utils import decode_robust, decode_robust_series
from .annotations import read_qupath_annotation
from parse import *
from .data import ImageData, BoundariesData, AnnotationData
from pandas.api.types import is_numeric_dtype
import warnings

def read_matrix(self, 
                read_cells: bool = True
                ):
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
                dapi_type: str = "focus",
                pattern_img_file: str = "{slide_id}__{region_id}__{image_name}__registered"
                ):
    # get available image keys in metadata
    img_keys = [elem for elem in self.metadata["images"] if elem.startswith("registered")]
    
    # get image files from keys
    img_files = [self.metadata["images"][k] for k in img_keys]
            
    # extract image names
    self.img_names = []
    for img_file in img_files:
        stem = Path(img_file).name.split(".")[0] # get stem of .ome.tif file
        
        # parse name
        img_file_parsed = parse(pattern_img_file, stem)
        self.img_names.append(img_file_parsed.named["image_name"])
    
    #self.img_names = [elem.split(".")[0].split("_")[1] for elem in img_files[1:]]
    
    # add information about dapi
    dapi_key = f"morphology_{dapi_type}_filepath"
    self.img_names = ["DAPI"] + self.img_names
    img_files = [self.metadata["images"][dapi_key]] + img_files
    img_keys = [dapi_key] + img_keys
    
    # load image
    self.images = ImageData(self.path, img_files, self.img_names)
    
def read_transcripts(self):
    # read transcripts
    self.transcripts = pd.read_parquet(self.path / self.transcript_filename)
    
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
                        ) -> pd.DataFrame:
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
        
