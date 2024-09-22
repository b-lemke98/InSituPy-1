import os
from copy import deepcopy
from pathlib import Path
from typing import Literal, Optional, Union
from uuid import uuid4

import pandas as pd

import insitupy
from insitupy._constants import LOAD_FUNCS
from insitupy._exceptions import ModalityNotFoundError
from insitupy.io.files import check_overwrite_and_remove_if_true
from insitupy.utils.utils import convert_to_list
from insitupy.utils.utils import textformat as tf


class InSituExperiment:
    def __init__(self):
        """
        Initialize an InSituExperiment object.

        """
        self._metadata = pd.DataFrame(columns=['uid', 'slide_id', 'sample_id'])
        self._data = []
        self._path = None

    def __repr__(self):
        """Provide a string representation of the InSituExperiment object.

        Returns:
            str: A string summarizing the InSituExperiment object.
        """
        num_samples = len(self._metadata)
        sample_summary = self._metadata.to_string(index=True, col_space=4)
        return (f"{tf.Bold}InSituExperiment{tf.ResetAll} with {num_samples} samples:\n"
                f"{sample_summary}")

    def __getitem__(self, key):
        """Retrieve a subset of the experiment.

        Args:
            key (int, slice, list, pd.Series): The index, slice, list, or boolean Series to retrieve.

        Returns:
            InSituExperiment: A new InSituExperiment object with the selected subset.
        """
        if isinstance(key, int):
            if key > (len(self)-1):
                raise IndexError(f"Index ({key}) is out of range {len(self)}.")
            key = slice(key, key + 1)
        elif isinstance(key, list) and all(isinstance(i, bool) for i in key):
            key = pd.Series(key)
        if isinstance(key, pd.Series) and key.dtype == bool:
            new_experiment = InSituExperiment()
            new_experiment._data = [d for d, k in zip(self._data, key) if k]
            new_experiment._metadata = self._metadata[key].reset_index(drop=True)
        else:
            new_experiment = InSituExperiment()
            new_experiment._data = self._data[key]
            new_experiment._metadata = self._metadata.iloc[key].reset_index(drop=True)

        # Disconnect object from save path
        new_experiment._path = None
        return new_experiment

    def __len__(self):
        """Return the number of datasets in the experiment.

        Returns:
            int: The number of datasets.
        """
        return len(self._data)

    @property
    def data(self):
        """Get the dataset dictionary.

        Returns:
            dict: A dictionary of datasets, where keys are sample IDs and values are Dataset objects.
        """
        return self._data

    @property
    def metadata(self):
        """Get the metadata DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing metadata.
        """
        return self._metadata.copy() # the copy prevents the metadata from being modified

    @property
    def path(self):
        """Return save path of the InSituExperiment object.

        Returns:
            str: Save path.
        """
        return self._path

    def add(self,
            data: Union[str, os.PathLike, Path, insitupy.InSituData],
            metadata: Optional[dict] = None
            ):
        """Add a dataset to the experiment and update metadata.

        Args:
            dataset (InSituData): A InSituData object to be added.

        Raises:
            TypeError: If the dataset is not an instance of the Dataset class.
        """
        # Check if the dataset is of the correct type
        try:
            data = Path(data)
        except TypeError:
            dataset = data
        else:
            dataset = insitupy.read_xenium(data)

        assert isinstance(dataset, insitupy.InSituData), "Loaded dataset is not an InSituData object."

        # # set a unique ID
        # dataset._set_uid()

        # Add the dataset to the data collection
        self._data.append(dataset)

        # Create a new DataFrame for the new metadata
        new_metadata = {
            'uid': str(uuid4()).split("-")[0],
            'slide_id': dataset.slide_id,
            'sample_id': dataset.sample_id
        }

        if metadata is not None:
            # add information from metadata argument
            new_metadata = metadata | new_metadata

        # convert to dataframe
        new_metadata = pd.DataFrame([new_metadata])

        # Concatenate the new metadata with the existing metadata
        self._metadata = pd.concat([self._metadata, new_metadata], axis=0, ignore_index=True)

    def copy(self):
        """Create a deep copy of the InSituExperiment object.

        Returns:
            InSituExperiment: A new InSituExperiment object that is a deep copy of the current object.
        """
        return deepcopy(self)

    def get(self,
            slide_id: str,
            sample_id: str
            ):
        """Retrieve a dataset by the combined key of slide_id and sample_id.

        Args:
            key (str): The combined key of slide_id and sample_id of the dataset to retrieve.

        Returns:
            Dataset: The dataset associated with the given key.

        Raises:
            KeyError: If the key does not exist in the data dictionary.
        """

        slide_mask = self._metadata["slide_id"] == slide_id
        sample_mask = self._metadata["sample_id"] == sample_id
        query_result = self._metadata[slide_mask & sample_mask]

        if len(query_result) == 1:
            index = query_result.index[0]
            return self.iget(index)
        elif len(query_result) == 0:
            print(f"No dataset with slide_id '{slide_id}' and sample_id '{sample_id}' found.")
        else:
            print("More than one possible dataset found. Query result:")
            print(query_result)


    def iget(self,
             index: int
             ):
        """Retrieve a dataset by its row position in the metadata DataFrame.

        Args:
            index (int): The row position of the dataset to retrieve.

        Returns:
            Dataset: The dataset associated with the given row position.

        Raises:
            IndexError: If the index is out of bounds.
        """

        return self._data[index]

    def load_all(self,
                 skip: Optional[str] = None,
                 ):
        for xd in self._data:
            print(xd.sample_id)
            for f in LOAD_FUNCS:
                if skip is None or skip not in f:
                    func = getattr(xd, f)
                    try:
                        func()
                    except ModalityNotFoundError as err:
                        print(err)

    def load_annotations(self):
        for xd in self._data:
            print(xd.sample_id)
            xd.load_annotations()

    def load_cells(self):
        for xd in self._data:
            print(xd.sample_id)
            xd.load_cells()

    def load_images(self,
                    names: Union[Literal["all", "nuclei"], str] = "all", # here a specific image can be chosen
                    nuclei_type: Literal["focus", "mip", ""] = "mip",
                    load_cell_segmentation_images: bool = True
                    ):

        for xd in self._data:
            print(xd.sample_id)
            xd.load_images(names=names,
                           nuclei_type=nuclei_type,
                           load_cell_segmentation_images=load_cell_segmentation_images)

    def load_regions(self):
        for xd in self._data:
            print(xd.sample_id)
            xd.load_regions()

    def load_transcripts(self,
                        transcript_filename: str = "transcripts.parquet"
                        ):
        for xd in self._data:
            print(xd.sample_id)
            xd.load_transcripts()

    def query(self, criteria: dict):
        """Query the experiment based on metadata criteria.

        Args:
            criteria (dict): A dictionary where keys are column names and values are lists of categories to select.

        Returns:
            InSituExperiment: A new InSituExperiment object with the selected subset.
        """
        mask = pd.Series([True] * len(self._metadata))
        for column, values in criteria.items():
            values = convert_to_list(values)
            if column in self._metadata.columns:
                mask &= self._metadata[column].isin(values)
            else:
                raise KeyError(f"Column '{column}' not found in metadata.")

        return self[mask]

    @classmethod
    def read(cls, path: Union[str, os.PathLike, Path]):
        """Read an InSituExperiment object from a specified folder.

        Args:
            path (Union[str, os.PathLike, Path]): The path to the folder where datasets are saved.

        Returns:
            InSituExperiment: A new InSituExperiment object with the loaded data.
        """
        path = Path(path)

        # Load metadata
        metadata_path = path / "metadata.csv"
        metadata = pd.read_csv(metadata_path, index_col=0)

        # Load each dataset
        data = []
        dataset_paths = sorted([elem for elem in Path("test").glob("*") if elem.is_dir()])
        #for i in range(len(metadata)):
        for dataset_path in dataset_paths:
            #dataset_path = path / f"{i}"
            dataset = insitupy.read_xenium(dataset_path)
            data.append(dataset)

        # Create a new InSituExperiment object
        experiment = cls()
        experiment._metadata = metadata
        experiment._data = data
        experiment._path = path

        return experiment

    def save(self):
        for xd in self._data:
            print(xd.sample_id)
            xd.save()


    def saveas(self, path: Union[str, os.PathLike, Path],
               overwrite: bool = False,
               verbose: bool = False):
        """Save all datasets to a specified folder.

        Args:
            path (Union[str, os.PathLike, Path]): The path to the folder where datasets will be saved.
        """
        # Create the main directory if it doesn't exist
        path = Path(path)

        # check overwrite
        check_overwrite_and_remove_if_true(path=path, overwrite=overwrite)

        print(f"Saving InSituExperiment to {str(path)}") if verbose else None

        # Iterate over the datasets and save each one in a numbered subfolder
        for index, dataset in enumerate(self._data):
            subfolder_path = path / f"{str(index).zfill(3)}"
            dataset.saveas(subfolder_path, verbose=False)

        # Optionally, save the metadata as a CSV file
        metadata_path = os.path.join(path, "metadata.csv")
        self._metadata.to_csv(metadata_path, index=True)

        print("Saved.") if verbose else None

