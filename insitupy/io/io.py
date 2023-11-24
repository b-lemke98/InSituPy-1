from pathlib import Path
import os
import pandas as pd
import geopandas
import requests
from tqdm import tqdm
from typing import Optional, Tuple, Union, List, Dict, Any, Literal
from geopandas.geodataframe import GeoDataFrame

# force geopandas to use shapely. Default in future versions of geopandas.
os.environ['USE_PYGEOS'] = '0' 


def read_qupath_geojson(file: Union[str, os.PathLike, Path]) -> pd.DataFrame:
    """
    Reads a QuPath-compatible GeoJSON file and transforms it into a flat DataFrame.

    Parameters:
    - file (Union[str, os.PathLike, Path]): The file path (as a string or pathlib.Path) of the QuPath GeoJSON file.

    Returns:
    pandas.DataFrame: A DataFrame with flattened columns including "name" and "color" extracted from the "classification" column.
    """
    # Read the GeoJSON file into a GeoDataFrame
    dataframe = geopandas.read_file(file)

    # Flatten the "classification" column into separate "name" and "color" columns
    dataframe["name"] = [elem["name"] for elem in dataframe["classification"]]
    dataframe["color"] = [elem["color"] for elem in dataframe["classification"]]

    # Remove the redundant "classification" column
    dataframe = dataframe.drop(["classification"], axis=1)

    # Return the transformed DataFrame
    return dataframe

def parse_geopandas(
    data: Union[GeoDataFrame, pd.DataFrame, dict,
                str, os.PathLike, Path]
    ):
    # check if the input is a path or a GeoDataFrame
    if isinstance(data, GeoDataFrame):
        df = data
    elif isinstance(data, pd.DataFrame) or isinstance(data, dict):
        df = GeoDataFrame(data, geometry=data["geometry"])
    else:
        # read annotations as GeoDataFrame
        data = Path(data)
        if data.suffix == ".geojson":
            df = read_qupath_geojson(file=data)
        else:
            raise ValueError(f"Unknown file extension: {data.suffix}. File is expected to be `.geojson` or `.parquet`.")
        
    return df
        


def write_qupath_geojson(dataframe: GeoDataFrame,
                         file: Union[str, os.PathLike, Path]
                         ):
    """
    Converts a GeoDataFrame with "name" and "color" columns into a QuPath-compatible GeoJSON-like format,
    adding a new "classification" column containing dictionaries with "name" and "color" entries.
    The modified GeoDataFrame is then saved to the specified GeoJSON file.

    Parameters:
    - dataframe (geopandas.GeoDataFrame): The input GeoDataFrame containing "name" and "color" columns.
    - file (Union[str, os.PathLike, Path]): The file path (as a string or pathlib.Path) where the GeoJSON data will be saved.
    """
    # Initialize an empty list to store dictionaries for each row
    classification_list = []

    # Iterate over rows in the GeoDataFrame
    for _, row in dataframe.iterrows():
        # Create a dictionary with "name" and "color" entries for each row
        classification_dict = {}
        for column in ["name", "color"]:
            classification_dict[column] = row[column]
        # Append the dictionary to the list
        classification_list.append(classification_dict)

    # Add a new "classification" column to the GeoDataFrame
    dataframe["classification"] = classification_list

    # Remove the original "name" and "color" columns
    dataframe = dataframe.drop(["name", "color"], axis=1)

    # Write the GeoDataFrame to a GeoJSON file
    dataframe.to_file(file, driver="GeoJSON")


def download_url(
    url: str,
    out_dir: Union[str, os.PathLike, Path] = ".",
    file_name: Optional[str] = None,
    chunk_size: int = 1024,
    overwrite: bool = False
    ) -> None:
    """
    Downloads a file from the specified URL and saves it to the given output directory.
    
    Code adapted from: https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51

    Args:
        url (str): The URL of the file to be downloaded.
        out_dir (Union[str, os.PathLike, Path], optional): The output directory where the downloaded file will be saved.
            Default is the current directory (".").
        file_name (str, optional): The name of the downloaded file. If not provided, the function will use the name
            from the URL. Default is None.
        chunk_size (int, optional): The size of the chunks in bytes to download the file. Default is 1024 bytes.
        overwrite (bool, optional): If True, the function will download the file even if it already exists in the
            output directory, overwriting the existing file. If False and the file exists, the function will skip
            the download. Default is False.

    Returns:
        None: This function does not return any value. The downloaded file is saved in the specified output directory.
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
    
    if outfile.exists():
        print(f"File {outfile} exists already. Download is skipped. To force download set `overwrite=True`.")
        return
    
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
            
