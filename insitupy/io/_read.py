from typing import Optional, Tuple, Union, List, Dict, Any, Literal
from pathlib import Path
import os
import scanpy as sc
import pandas as pd
from ..utils.utils import decode_robust_series, convert_to_list
from parse import *
from ..utils.data import ImageData, BoundariesData, AnnotationData
from ..utils.exceptions import ModalityNotFoundError
from pandas.api.types import is_numeric_dtype
import warnings

def read_matrix(self, 
                read_cells: bool = True
                ):
    if self.from_xeniumdata:
        # check if matrix data is stored in this XeniumData
        if "matrix" not in self.xd_metadata:
            raise ModalityNotFoundError(modality="matrix")
        
        # read matrix data
        print("Reading matrix...", flush=True)
        self.matrix = sc.read(self.path / self.xd_metadata["matrix"])
    else:
        print("Reading matrix...", flush=True)
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
        # check if matrix data is stored in this XeniumData
        if "images" not in self.xd_metadata:
            raise ModalityNotFoundError(modality="images")
        
        # get file paths and names
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
    print("Reading images...", flush=True)
    self.images = ImageData(self.path, img_files, img_names)
    
    
def read_transcripts(self,
                     transcript_filename: str = "transcripts.parquet"
                     ):
    if self.from_xeniumdata:
        # check if matrix data is stored in this XeniumData
        if "transcripts" not in self.xd_metadata:
            raise ModalityNotFoundError(modality="transcripts")
        
        # read transcripts
        print("Reading transcripts...", flush=True)
        self.transcripts = pd.read_parquet(self.path / self.xd_metadata["transcripts"])
    else:
        # read transcripts
        print("Reading transcripts...", flush=True)
        self.transcripts = pd.read_parquet(self.path / transcript_filename)
        
        # decode columns
        self.transcripts = self.transcripts.apply(lambda x: decode_robust_series(x), axis=0)
        
        # convert coordinates into pixel coordinates
        coord_cols = ["x_location", "y_location", "z_location"]
        self.transcripts[coord_cols] = self.transcripts[coord_cols].apply(lambda x: x / self.metadata["pixel_size"])
        
def read_boundaries(self,
                    files: List[str] = ["cell_boundaries.parquet", "nucleus_boundaries.parquet"],
                    labels: List[str] = ["cells", "nuclei"]
                    ):
    if self.from_xeniumdata:
        # check if matrix data is stored in this XeniumData
        if "boundaries" not in self.xd_metadata:
            raise ModalityNotFoundError(modality="boundaries")
        
        # get path and names of boundary files
        labels = self.xd_metadata["boundaries"].keys()
        files = [self.xd_metadata["boundaries"][n] for n in labels]

    # convert arguments to lists
    labels = convert_to_list(labels)
    files = convert_to_list(files)
        
    # read boundaries data
    print("Reading boundaries...", flush=True)
    self.boundaries = BoundariesData(path=self.path,
                                    files=files,
                                    labels=labels,
                                    pixel_size=self.metadata["pixel_size"]
                                    )
    
def read_annotations(self,
                     annotation_dir: Union[str, os.PathLike, Path] = None, # "../annotations",
                     suffix: str = ".geojson",
                     pattern_annotation_file: str = "annotation-{slide_id}__{sample_id}__{name}"
                     ):
    if self.from_xeniumdata:
        # check if matrix data is stored in this XeniumData
        if "annotations" not in self.xd_metadata:
            raise ModalityNotFoundError(modality="annotations")
        
        # get path and names of annotation files
        labels = self.xd_metadata["annotations"].keys()
        files = [self.path / self.xd_metadata["annotations"][n] for n in labels]
        
    else:
        if annotation_dir is None:
            raise ModalityNotFoundError(modality="annotations")
        else:
            # convert to Path
            annotation_dir = Path(annotation_dir)
            
            # check if the annotation path exists. If it does not, first assume that it is a relative path and check that.
            if not annotation_dir.is_dir():
                annotation_dir = Path(os.path.normpath(self.path / annotation_dir))
                if not annotation_dir.is_dir():
                    raise FileNotFoundError(f"`annot_path` {annotation_dir} is neither a direct path nor a relative path.")
            
            # get list annotation files that match the current slide id and sample id
            files = []
            labels = []
            for file in annotation_dir.glob(f"*{suffix}"):
                if self.slide_id in str(file.stem) and (self.sample_id in str(file.stem)):
                    parsed = parse(pattern_annotation_file, file.stem)
                    labels.append(parsed.named["name"])
                    files.append(file)
            
    print("Reading annotations...", flush=True)
    self.annotations = AnnotationData(annot_files=files, annot_labels=labels)
            
def read_all(self, verbose: bool = True):
    # extract read functions
    read_funcs = [elem for elem in dir(self) if elem.startswith("read_")]
    read_funcs = [elem for elem in read_funcs if elem != "read_all"]
    
    for f in read_funcs:
        # if verbose: 
        #     print(f"Running {f}()", flush=True)
        func = getattr(self, f)
        try:
            func()
        except ModalityNotFoundError as err:
            print(err)
            
        