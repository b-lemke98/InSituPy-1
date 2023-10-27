from typing import Optional, Tuple, Union, List, Dict, Any, Literal
from pathlib import Path
import os
from ..images.io import write_ome_tiff
 
def save(self,
         path: Union[str, os.PathLike, Path]
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
            filename = Path(metadata["file"]).name
            img = getattr(self.images, n)[0]
            photometric = 'rgb' if metadata['rgb'] else 'minisblack'
            write_ome_tiff(img_path / filename, img, photometric=photometric)