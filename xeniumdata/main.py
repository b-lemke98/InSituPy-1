import json
from typing import Optional, Tuple, Union, List, Dict, Any, Literal
from pathlib import Path
import os
import scanpy as sc
import pandas as pd
from dask_image.imread import imread
from .utils import textformat as tf

def read_xenium_metadata(
    path: Union[str, os.PathLike, Path],
    metadata_filename: str = "experiment.xenium"
    ) -> dict:
    '''
    Function to read the xenium metadata file which usually is in the xenium output folder of one region.
    '''
    # load metadata file
    metapath = path / metadata_filename
    with open(metapath, "r") as metafile:
        metadata = json.load(metafile)
        
    return metadata

class XeniumData:
    '''
    XeniumData object to read Xenium in situ data in a structured way.
    '''
    def __init__(self, 
                 path: Union[str, os.PathLike, Path],
                 metadata_filename: str = "experiment_modified.xenium",
                 transcript_filename: str = "transcripts.parquet"
                 ):
        self.path = Path(path)
        self.metadata_filename = metadata_filename
        self.transcript_filename = transcript_filename
        self.metadata = read_xenium_metadata(path, metadata_filename=metadata_filename)
        
    def __repr__(self):
        repr = (
            f"{tf.BOLD+tf.RED} XeniumData{tf.END}\n" 
            f"{tf.BOLD}Data path:{tf.END} {self.path.parent}\n"
            f"{tf.BOLD}Data folder:{tf.END} {self.path.name}\n"
            f"{tf.BOLD}Metadata file:{tf.END} {self.metadata_filename}"            
        )
        
        if hasattr(self, "images"):
            images_repr = self.images.__repr__()
            repr = (
                #repr + f"\n{color.BOLD}Images:{color.END} "
                repr + f"\n\t{tf.RARROWHEAD} " + images_repr.replace("\n", "\n\t   ")
            )
            
        if hasattr(self, "matrix"):
            matrix_repr = self.matrix.__repr__()
            repr = (
                repr + f"\n\t{tf.RARROWHEAD+tf.GREEN+tf.BOLD} matrix{tf.END}\n\t   " + matrix_repr.replace("\n", "\n\t   ")
            )
        
        if hasattr(self, "transcripts"):
            #trans_repr = str(type(self.transcripts)).split(" ")[1].rstrip(">") + \
            trans_repr = f"DataFrame with shape {self.transcripts.shape[0]} x {self.transcripts.shape[1]}"
            
            repr = (
                repr + f"\n\t{tf.RARROWHEAD+tf.CYAN+tf.BOLD} transcripts{tf.END}\n\t   " + trans_repr
            )
        return repr
    
    def read_matrix(self, 
                    read_cells: bool = True
                    ):
        cf_zarr_path = self.path / self.metadata["xenium_explorer_files"]["cell_features_zarr_filepath"]
        cf_h5_path = cf_zarr_path.parent / cf_zarr_path.name.replace(".zarr.zip", ".h5")

        # read matrix data
        self.matrix = sc.read_10x_h5(cf_h5_path)
        
        if read_cells:
            # read cell information
            cells_zarr_path = self.path / self.metadata["xenium_explorer_files"]["cells_zarr_filepath"]
            cells_parquet_path = cells_zarr_path.parent / cells_zarr_path.name.replace(".zarr.zip", ".parquet")
            cells = pd.read_parquet(cells_parquet_path)
            
            # transform cell ids from bytes to str
            cells = cells.set_index("cell_id")
            cells.index = [elem.decode() for elem in cells.index]
            
            # add information to anndata observations
            self.matrix.obs = pd.merge(left=self.matrix.obs, right=cells, left_index=True, right_index=True)
        
    def read_images(self,
                    dapi_type: str = "focus"
                    ):
        # get available image keys in metadata
        dapi_key = f"morphology_{dapi_type}_filepath"
        img_keys = [elem for elem in self.metadata["images"] if elem.startswith("registered")]
        img_keys = [dapi_key] + img_keys
        
        # get image files from keys
        img_files = [self.metadata["images"][k] for k in img_keys]
                
        # extract image names
        self.img_names = [elem.split(".")[0].split("_")[1] for elem in img_files[1:]]
        self.img_names = ["DAPI"] + self.img_names
        
        # load image
        self.images = ImageData(self.path, img_files, self.img_names, dapi_type)
        
    def read_transcripts(self):
        # read transcripts
        self.transcripts = pd.read_parquet(self.path / self.transcript_filename)
        
        
class ImageData:
    '''
    Object to read and load images.
    '''
    def __init__(self, path, img_files, img_names, dapi_type):
        
        self.img_files = img_files
        self.img_names = img_names
        
        self.img_shapes = []
        for n, f in zip(self.img_names, self.img_files):
            # load images
            img = imread(path / f)[0]
            setattr(self, n, img)
            self.img_shapes.append(img.shape)
        
    def __repr__(self):
        repr_strings = [f"{tf.BOLD + a + tf.END}\t{b}" for a,b in zip(self.img_names, self.img_shapes)]
        s = "\n".join(repr_strings)
        repr = f"{tf.BLUE+tf.BOLD}images{tf.END}\n{s}"
        return repr
    
    def load(self, 
             which: Union[List[str], str] = "all"
             ):
        '''
        Load images into memory.
        '''
        if which == "all":
            which = self.img_names
            
        # make sure which is a list
        which = [which] if isinstance(which, str) else list(which)
        for n in which:
            img_loaded = getattr(self, n).compute()
            setattr(self, n, img_loaded)
