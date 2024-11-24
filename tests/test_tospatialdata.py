import pytest
from pathlib import Path
from insitupy import read_xenium
from spatialdata import read_zarr
import numpy as np
import filecmp
import hashlib
import os
import shutil

DATA_PATH = Path("tests/data/breast_cropped")
TMP_PATH = Path("tmp_sdata.zarr")
REFERENCE_PATH = Path("tests/data/sdata_reference.zarr")
image_y = 81
image_x = 67
n_cells = 4
n_transripts = 410
n_genes = 313
n_images = 3
image_names = ["nuclei", "DAPI", "HE"]


def test_project_loading():
    xd = read_xenium(DATA_PATH)
    xd.load_all()

    assert len(xd.images.metadata) == n_images
    assert len(xd.images.get("nuclei")) == 8
    assert xd.images.get("nuclei")[0].shape == (image_y, image_x)

    assert len(xd.cells.boundaries.metadata) == 2
    assert xd.cells.boundaries.cellular[0].shape == (image_y, image_x)
    assert xd.cells.boundaries.nuclear[0].shape == (image_y, image_x)

    assert xd.transcripts.shape == (n_transripts, 7)
    assert xd.cells.matrix.shape == (n_cells, n_genes)


@pytest.mark.parametrize("levels", [1, 3])
def test_tospatialdata(levels: int):
    xd = read_xenium(DATA_PATH)
    xd.load_all()

    sdata = xd.to_spatialdata(levels)

    sdata_ref = read_zarr(REFERENCE_PATH)

    assert len(sdata.images) == n_images
    assert len(sdata.images["nuclei"].groups) == levels + 2
    assert sdata.images["nuclei"]["scale0"].dims == {'c': 1, 'y': image_y, 'x': image_x}
    assert sdata.images["DAPI"]["scale0"].dims == {'c': 1, 'y': image_y, 'x': image_x}
    assert sdata.images["HE"]["scale0"].dims == {'c': 3, 'y': image_y, 'x': image_x}
        

    assert len(sdata.labels) == 2
    assert len(sdata.labels["cellular"].groups) == levels + 2
    assert len(sdata.labels["nuclear"].groups) == levels + 2
    assert sdata.labels["cellular"]["scale0"].dims == {'y': image_y, 'x': image_x}
    assert sdata.labels["nuclear"]["scale0"].dims == {'y': image_y, 'x': image_x}


    if levels == 1:
        for name in image_names:
            assert sdata.images[name]["scale0"].equals(sdata_ref.images[name]["scale0"])

        assert sdata.labels["cellular"]["scale0"].equals(sdata_ref.labels["cellular"]["scale0"])
        assert sdata.labels["nuclear"]["scale0"].equals(sdata_ref.labels["nuclear"]["scale0"])

    df1 = sdata.points["transcripts"].compute()
    df2 = sdata_ref.points["transcripts"].compute()
    df1['feature_name'] = df1['feature_name'].astype(str)
    df2['feature_name'] = df2['feature_name'].astype(str)
    assert df1.equals(df2)

    assert sdata.shapes["cell_circles"].equals(sdata.shapes["cell_circles"])

    assert len(sdata.tables) == 1
    from anndata.tests.helpers import assert_adata_equal
    assert_adata_equal(sdata.tables["table"], sdata_ref.tables["table"], exact=False)

""" def test_tospatialdata_files():
    xd = read_xenium(DATA_PATH)
    xd.load_all()

    sdata = xd.to_spatialdata(levels=1)
    sdata.write(TMP_PATH, overwrite=True)

    def md5_file(filename):
        # Create a hash object
        h = hashlib.md5()

        # Open file for reading in binary mode
        with open(filename, 'rb') as file:
            # Read and update hash string value in blocks of 4K
            for chunk in iter(lambda: file.read(4096), b""):
                h.update(chunk)

        # Return the hex representation of the digest
        return h.hexdigest()

    def compare_files(file1, file2):
        return md5_file(file1) == md5_file(file2)

    def compare_directories(dir1, dir2):
        # Compare the directories
        comparison = filecmp.dircmp(dir1, dir2)
        diff = []

        # Compare the contents of the common files
        for file_name in comparison.common_files:
            file1 = os.path.join(dir1, file_name)
            file2 = os.path.join(dir2, file_name)
            if not compare_files(file1, file2):
                diff.append(f"The files {file1} and {file2} are different.")

        # Print the different files
        for file_name in comparison.diff_files:
            diff.append(f"Different file: {file_name}")

        # Print files only in dir1
        for file_name in comparison.left_only:
            diff.append(f"File only in {dir1}: {file_name}")

        # Print files only in dir2
        for file_name in comparison.right_only:
            diff.append(f"File only in {dir2}: {file_name}")

        # Recursively compare subdirectories
        for subdir in comparison.common_dirs:
            subdir1 = os.path.join(dir1, subdir)
            subdir2 = os.path.join(dir2, subdir)
            if 'binned_expression' not in subdir1 and 'binned_expression' not in subdir2:
                compare_directories(subdir1, subdir2)
        return diff
    
    def delete_directory(directory_path):
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
            print(f"Directory '{directory_path}' has been deleted.")
        else:
            print(f"Directory '{directory_path}' does not exist.")
    
    assert len(compare_directories(TMP_PATH, "tests/data/sdata.zarr")) == 0
    delete_directory(TMP_PATH) """

