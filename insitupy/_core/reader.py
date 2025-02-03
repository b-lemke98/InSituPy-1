import os
from os.path import abspath
from pathlib import Path
from typing import Literal, Optional, Union
from uuid import uuid4

from parse import *

from insitupy import __version__
from insitupy._constants import ISPY_METADATA_FILE
from insitupy._core.insitudata import InSituData
from insitupy.io.files import read_json


def read(
    path: Union[str, os.PathLike, Path],
    mode: Literal["insitupy", "xenium"] = "insitupy"
):
    if mode == "insitupy":
        return _read_insitupy(
            path=path
        )

    elif mode == "xenium":
        return _read_xenium(
            path=path
            )
    else:
        raise ValueError(f"Unknown mode {mode}.")

def _read_insitupy(
    path: Union[str, os.PathLike, Path],
    ) -> InSituData:
    metadata_filename: str = "experiment.xenium"
    path = Path(path) # make sure the path is a pathlib path
    assert (path / ISPY_METADATA_FILE).exists(), "No insitupy metadata file found."
    # read InSituData metadata
    insitupy_metadata_file = path / ISPY_METADATA_FILE
    metadata = read_json(insitupy_metadata_file)

    # retrieve slide_id and sample_id
    slide_id = metadata["slide_id"]
    sample_id = metadata["sample_id"]

    # save paths of this project in metadata
    metadata["path"] = abspath(path).replace("\\", "/")
    metadata["metadata_file"] = ISPY_METADATA_FILE

    data = InSituData(path=path,
                        metadata=metadata,
                        slide_id=slide_id,
                        sample_id=sample_id,
                        from_insitudata=True,
                        )

    return data

def _read_xenium(
    path: Union[str, os.PathLike, Path],
    ) -> InSituData:
    metadata_filename: str = "experiment.xenium"
    path = Path(path) # make sure the path is a pathlib path

    # initialize the metadata dict
    metadata = {}
    metadata["data"] = {}
    metadata["history"] = {}
    metadata["history"]["cells"] = []
    metadata["history"]["annotations"] = []
    metadata["history"]["regions"] = []

    # check if path exists
    if not path.is_dir():
        raise FileNotFoundError(f"No such directory found: {str(path)}")

    # save paths of this project in metadata
    metadata["path"] = abspath(path).replace("\\", "/")
    metadata["metadata_file"] = metadata_filename

    # read metadata
    metadata["method_params"] = read_json(path / metadata_filename)

    # get slide id and sample id from metadata
    slide_id = metadata["method_params"]["slide_id"]
    sample_id = metadata["method_params"]["region_name"]

    # initialize the uid section
    metadata["uids"] = [str(uuid4())]

    # add method
    metadata["method"] = "Xenium"

    data = InSituData(path=path,
                        metadata=metadata,
                        slide_id=slide_id,
                        sample_id=sample_id,
                        from_insitudata=False,
                        )

    return data
