import requests
from tqdm import tqdm
from typing import Optional, Tuple, Union, List, Dict, Any, Literal
from pathlib import Path
import os

def download_url(
    url: str,
    out_dir: Union[str, os.PathLike, Path] = ".",
    file_name: Optional[str] = None,
    chunk_size=1024
    ):
    """Downloads a file from the specified URL and saves it to the given file path.
    
    Code adapted from: https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51

    Args:
        url (str): The URL from which the file will be downloaded.
        out_dir (Union[str, os.PathLike, Path], optional): The directory where the downloaded file will be saved in. 
                                                           Defaults to ".".
        file_name (Optional[str], optional): The name (without suffix) under which the downloaded file is saved. If None, the name is
                                             is inferred from the URL. Defaults to None.
        chunk_size (int, optional): The size of each chunk for streaming the download. 
                                    Defaults to 1024 bytes.
    """
    # create output directory if necessary
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # check which file name to use
    suffix = f".{Path(url).name.split('.', maxsplit=1)[-1]}" # get suffix (robustly against multiple dots like .ome.tif)
    if file_name is None:
        file_name = Path(url).stem
    
    # create path for output file
    outfile = out_dir / (file_name + suffix)
    
    # request content from URL
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    
    # write to file
    with open(str(outfile), 'wb') as file, tqdm(
        desc=str(outfile),
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)
            
