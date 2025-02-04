import os
from os.path import abspath
from pathlib import Path
from typing import Literal, Optional, Union
from uuid import uuid4
from warnings import warn

import pandas as pd
from parse import *

from insitupy import __version__
from insitupy._constants import ISPY_METADATA_FILE
from insitupy._core._xenium import (_read_binned_expression,
                                    _read_boundaries_from_xenium,
                                    _read_matrix_from_xenium,
                                    _restructure_transcripts_dataframe)
from insitupy._core.insitudata import InSituData
from insitupy._exceptions import InvalidXeniumDirectory
from insitupy.io.files import read_json
from insitupy.utils.utils import convert_to_list

from .dataclasses import AnnotationsData, CellData, ImageData, RegionsData


def read_xenium(
    path: Union[str, os.PathLike, Path],
    # names: Union[Literal["all", "nuclei"], str] = "all", # here a specific image can be chosen
    nuclei_type: Literal["focus", "mip", ""] = "mip",
    load_cell_segmentation_images: bool = True,
    verbose: bool = True
    ) -> InSituData:

    path = Path(path) # make sure the path is a pathlib path
    metadata_filename: str = "experiment.xenium"

    if not (path / metadata_filename).exists():
        raise InvalidXeniumDirectory(directory=path)

    # INITIALIZE INSITUDATA
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
                        from_insitudata=False)

    # LOAD CELLS
    if verbose:
        print("Loading cells...", flush=True)
    pixel_size = data.metadata["method_params"]["pixel_size"]
    # read celldata
    matrix = _read_matrix_from_xenium(path=data.path)
    boundaries = _read_boundaries_from_xenium(path=data.path, pixel_size=pixel_size)
    data.cells = CellData(matrix=matrix, boundaries=boundaries)

    try:
        # read binned expression
        arr = _read_binned_expression(path=data.path, gene_names_to_select=data.cells.matrix.var_names)
        data.cells.matrix.varm["binned_expression"] = arr
    except ValueError:
        warn("Loading of binned expression did not work. Skipped it.")
        pass

    # LOAD IMAGES
    if verbose:
        print("Loading images...", flush=True)
    nuclei_file_key = f"morphology_{nuclei_type}_filepath"

    # In v2.0 the "mip" image was removed due to better focusing of the machine.
    # For <v2.0 the function still tries to retrieve the "mip" image but in case this is not found
    # it will retrieve the "focus" image
    if nuclei_type == "mip" and nuclei_file_key not in data.metadata["method_params"]["images"].keys():
        warn(
            f"Nuclei image type '{nuclei_type}' not found. Used 'focus' instead. This is the normal behavior for data analyzed with Xenium Ranger >=v2.0",
            UserWarning, stacklevel=2
                )

        nuclei_type = "focus"
        nuclei_file_key = f"morphology_{nuclei_type}_filepath"

    # if names == "nuclei":
    img_keys = [nuclei_file_key]
    img_names = ["nuclei"]
    # else:
    #     # get available keys for registered images in metadata
    #     img_keys = [elem for elem in data.metadata["method_params"]["images"] if elem.startswith("registered")]

    #     # extract image names from keys and add nuclei
    #     img_names = ["nuclei"] + [elem.split("_")[1] for elem in img_keys]

    #     # add dapi image key
    #     img_keys = [nuclei_file_key] + img_keys

    #     if names != "all":
    #         # make sure keys is a list
    #         names = convert_to_list(names)
    #         # select the specified keys
    #         mask = [elem in names for elem in img_names]
    #         img_keys = [elem for m, elem in zip(mask, img_keys) if m]
    #         img_names = [elem for m, elem in zip(mask, img_names) if m]

    # get path of image files
    img_files = [data.metadata["method_params"]["images"][k] for k in img_keys]

    if load_cell_segmentation_images:
        # get cell segmentation images if available
        if "morphology_focus/" in data.metadata["method_params"]["images"][nuclei_file_key]:
            seg_files = ["morphology_focus/morphology_focus_0001.ome.tif",
                            "morphology_focus/morphology_focus_0002.ome.tif",
                            "morphology_focus/morphology_focus_0003.ome.tif"
                            ]
            seg_names = ["cellseg1", "cellseg2", "cellseg3"]

            # check which segmentation files exist and append to image list
            seg_file_exists_list = [(data.path / f).is_file() for f in seg_files]
            #print(seg_file_exists_list)
            img_files += [f for f, exists in zip(seg_files, seg_file_exists_list) if exists]
            img_names += [n for n, exists in zip(seg_names, seg_file_exists_list) if exists]

    # create imageData object
    img_paths = [data.path / elem for elem in img_files]

    if data.images is None:
        data.images = ImageData(img_paths, img_names)
    else:
        for im, n in zip(img_paths, img_names):
            data.images.add_image(im, n, overwrite=False, verbose=True)

    # LOAD TRANSCRIPTS
    transcript_filename = "transcripts.parquet"
    if verbose:
        print("Loading transcripts...", flush=True)
    transcript_dataframe = pd.read_parquet(data.path / transcript_filename)

    data.transcripts = _restructure_transcripts_dataframe(transcript_dataframe)

    return data
