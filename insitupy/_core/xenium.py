import os
from os.path import abspath
from pathlib import Path
from typing import Optional, Union
from uuid import uuid4

from parse import *

from insitupy import __version__
from insitupy._constants import ISPY_METADATA_FILE
from insitupy._core.insitudata import InSituData
from insitupy.io.files import read_json


def read_xenium(
    path: Union[str, os.PathLike, Path],
    metadata_filename: str = "experiment.xenium",
) -> InSituData:
        """_summary_

        Args:
            path (Union[str, os.PathLike, Path]): _description_
            pattern_xenium_folder (str, optional): _description_. Defaults to "output-{ins_id}__{slide_id}__{sample_id}".
            matrix (Optional[AnnData], optional): _description_. Defaults to None.

        Raises:
            FileNotFoundError: _description_
        """
        path = Path(path) # make sure the path is a pathlib path
        dim = None # dimensions of the dataset
        from_insitudata = False  # flag indicating from where the data is read
        if (path / ISPY_METADATA_FILE).exists():
            # read InSituData metadata
            xd_metadata_file = path / ISPY_METADATA_FILE
            metadata = read_json(xd_metadata_file)

            # retrieve slide_id and sample_id
            slide_id = metadata["slide_id"]
            sample_id = metadata["sample_id"]

            # save paths of this project in metadata
            metadata["path"] = abspath(path).replace("\\", "/")
            metadata["metadata_file"] = ISPY_METADATA_FILE

            # set flag for InSituData
            from_insitudata = True
        else:
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

        # # add method to metadata
        # metadata["method"] = "Xenium"

        data = InSituData(path=path,
                          metadata=metadata,
                          method="Xenium",
                          slide_id=slide_id,
                          sample_id=sample_id,
                          from_insitudata=from_insitudata,
                          )

        return data
