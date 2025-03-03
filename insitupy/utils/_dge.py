from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from anndata import AnnData

from insitupy._core._checks import _check_assignment
from insitupy._core.insitudata import InSituData


# Define the function
def _check_string_in_assignment(entry, string_to_check):
    split_list = entry.split(" & ")
    return string_to_check in split_list

def _select_data_for_dge(
    data: InSituData,
    annotation_tuple: Optional[Tuple[str, str]] = None,
    cell_type_tuple: Optional[Tuple[str, str]] = None,
    region_tuple: Optional[Tuple[str, str]] = None,
    force_assignment: bool = False,
    ) -> AnnData:

    # extract anndata object
    adata = data.cells.matrix.copy()

    ### REGIONS
    if region_tuple is not None:
        # extract infos from tuple

        # assign region
        _check_assignment(data=data, key=region_tuple[0], force_assignment=force_assignment, modality="regions")

        # select only one region
        s = adata.obsm["regions"][region_tuple[0]]
        region_mask = s.apply(_check_string_in_assignment, string_to_check=region_tuple[1])
        if not np.any(region_mask):
            raise ValueError(f"Region '{region_tuple[1]}' not found in key '{region_tuple[0]}'.")

        print(f"Restrict analysis to region '{region_tuple[1]}' from key '{region_tuple[0]}'.", flush=True)
        adata = adata[region_mask].copy()

    ### ANNOTATIONS
    if annotation_tuple is not None:
        # check if the annotations need to be assigned first
        _check_assignment(data=data, key=annotation_tuple[1], force_assignment=force_assignment, modality="annotations")

        # create mask for filtering
        s = adata.obsm["annotations"][annotation_tuple[0]]
        annot_mask = s.apply(_check_string_in_assignment, string_to_check=annotation_tuple[1])
        if not np.any(annot_mask):
            raise ValueError(f"annotation_name '{annotation_tuple[1]}' not found under annotation_key '{annotation_tuple[0]}'.")

        # do filtering
        print(f"Restrict analysis to annotation '{annotation_tuple[1]}' from key '{annotation_tuple[0]}'.", flush=True)
        adata = adata[annot_mask].copy()

    ### CELL TYPES
    if cell_type_tuple is not None:
        cell_type_mask = adata.obs[cell_type_tuple[0]] == cell_type_tuple[1]

        if not np.any(cell_type_mask):
            raise ValueError(f"Cell type '{cell_type_tuple[1]}' not found in .obs column '{cell_type_tuple[0]}'.")

        print(f"Restrict analysis to cell type '{cell_type_tuple[1]}' from .obs column '{cell_type_tuple[0]}'.", flush=True)
        adata = adata[cell_type_mask].copy()

    return adata