from typing import Optional, Tuple, Union, List, Dict, Any, Literal
from pathlib import Path
import os
from ..images.io import write_ome_tiff
 
def save(self,
         path: Union[str, os.PathLike, Path],
         overwrite: bool = False
         ):
    '''
    Function to save the XeniumData object.
    
    Args:
        path: Path to save the data to.
    '''
    # create output directory if it does not exist yet
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # save images
    if hasattr(self, "images"):
        img_path = (path / "images")
        img_path.mkdir(parents=True, exist_ok=True) # create image directory
        
        for n, metadata in self.images.metadata.items():
            # extract image
            img = getattr(self.images, n)[0]
            
            # get file name for saving
            filename = Path(metadata["file"]).name
            
            # retrieve metadata for saving
            photometric = 'rgb' if metadata['rgb'] else 'minisblack'
            axes = metadata['axes']
            
            # retrieve OME metadata
            ome_meta_to_retrieve = ["SignificantBits", "PhysicalSizeX", "PhysicalSizeY", "PhysicalSizeXUnit", "PhysicalSizeYUnit"]
            pixel_meta = metadata["OME"]["Image"]["Pixels"]
            selected_metadata = {key: pixel_meta[key] for key in ome_meta_to_retrieve if key in pixel_meta}
            
            write_ome_tiff(img_path / filename, img, photometric=photometric, axes=axes, metadata=selected_metadata, overwrite=overwrite)
            
    # save matrix
    if hasattr(self, "matrix"):
        mtx_path = (path / "matrix")
        mtx_path.mkdir(parents=True, exist_ok=True) # create directory
        self.matrix.write(mtx_path / "matrix.h5ad")
        
    # save transcripts
    if hasattr(self, "transcripts"):
        trans_path = (path / "transcripts")
        trans_path.mkdir(parents=True, exist_ok=True) # create directory
        self.transcripts.to_parquet(trans_path / "transcripts.parquet")
        
    # save boundaries
    if hasattr(self, "boundaries"):
        bound_path = (path / "boundaries")
        bound_path.mkdir(parents=True, exist_ok=True) # create directory
        
        for n in ["cells", "nuclei"]:
            bounddf = getattr(self.boundaries, n)        
            bounddf.to_parquet(bound_path / f"{n}.parquet")
            
    # save annotations
    if hasattr(self, "annotations"):
        annot_path = (path / "annotations")
        annot_path.mkdir(parents=True, exist_ok=True) # create directory
        
        for n in self.annotations.labels:
            annotdf = getattr(self.annotations, n)        
            annotdf.to_parquet(annot_path / f"{n}.parquet")