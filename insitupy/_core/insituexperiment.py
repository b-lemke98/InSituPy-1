import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd

import insitupy
from insitupy._constants import LOAD_FUNCS
from insitupy.utils.utils import textformat as tf


class InSituExperiment:
    def __init__(self):
        """
        Initialize an InSituExperiment object.

        """
        self._metadata = pd.DataFrame(columns=['slide_id', 'sample_id'])
        self._data = []

    def __repr__(self):
        """Provide a string representation of the InSituExperiment object.

        Returns:
            str: A string summarizing the InSituExperiment object.
        """
        num_samples = len(self._metadata)
        sample_summary = self._metadata.to_string(index=True, col_space=4)
        return (f"{tf.Bold}InSituExperiment{tf.ResetAll} with {num_samples} samples:\n"
                f"{sample_summary}")

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
        return self._metadata

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

        # Use the combination of slide_id and sample_id as the key
        #key = self._key_pattern.format(slide_id=dataset.slide_id, sample_id=dataset.sample_id)

        # Add the dataset to the data collection
        #self._data[key] = dataset
        self._data.append(dataset)

        # Create a new DataFrame for the new metadata
        new_metadata = {
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
        # key = self._key_pattern.format(slide_id=slide_id, sample_id=sample_id)

        # if key not in self._data:
        #     raise KeyError(f"Dataset with key '{key}' not found.")
        # return self._data[key]

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
        # if index < 0 or index >= len(self._metadata):
        #     raise IndexError("Index out of bounds.")
        # slide_id, sample_id = self._metadata.iloc[index][["slide_id", "sample_id"]]
        # return self.get(slide_id=slide_id, sample_id=sample_id)
        return self._data[index]

    def load_all(self,
                 skip: Optional[str] = None,
                 ):
        for f in LOAD_FUNCS:
            if skip is None or skip not in f:
                func = getattr(self, f)
                try:
                    func()
                except ModalityNotFoundError as err:
                    print(err)

    def save(self, path: Union[str, os.PathLike, Path], overwrite: bool = False):
        """Save all datasets to a specified folder.

        Args:
            path (Union[str, os.PathLike, Path]): The path to the folder where datasets will be saved.
        """
        # Create the main directory if it doesn't exist
        path = Path(path)
        path.mkdir(exist_ok=True)

        # check overwrite
        check_overwrite_and_remove_if_true(path=path, overwrite=overwrite)

        # Iterate over the datasets and save each one in a numbered subfolder
        for index, dataset in enumerate(self._data):
            subfolder_path = path / str(index)
            dataset.saveas(subfolder_path)

        # Optionally, save the metadata as a CSV file
        metadata_path = os.path.join(path, "metadata.csv")
        self._metadata.to_csv(metadata_path, index=True)

