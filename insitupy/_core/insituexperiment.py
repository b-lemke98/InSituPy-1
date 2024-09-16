from pathlib import Path
from typing import Optional

import pandas as pd

import insitupy


class InSituExperiment:
    def __init__(self):
        """
        Initialize an InSituExperiment object.

        Args:
            patient_id (str): Unique identifier for the patient.
            disease (str): Disease associated with the experiment.
            age (int): Age of the patient.
            sex (str): Sex of the patient.
        """
        self._metadata = pd.DataFrame(columns=['sample_id', 'slide_id'])
        self._data = {}
        self._key_pattern = "{slide_id}__{sample_id}"

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
            data: insitupy.InSituData,
            metadata: Optional[dict] = None
            ):
        """Add a dataset to the experiment and update metadata.

        Args:
            dataset (Dataset): A dataset object to be added.

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
        key = self._key_pattern.format(slide_id=dataset.slide_id, sample_id=dataset.sample_id)

        # Add the dataset to the data dictionary
        self._data[key] = dataset

        # Create a new DataFrame for the new metadata
        new_metadata = {
            'sample_id': dataset.sample_id,
            'slide_id': dataset.slide_id
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
        key = self._key_pattern.format(slide_id=slide_id, sample_id=sample_id)

        if key not in self._data:
            raise KeyError(f"Dataset with key '{key}' not found.")
        return self._data[key]

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
        if index < 0 or index >= len(self._metadata):
            raise IndexError("Index out of bounds.")
        slide_id, sample_id = self._metadata.iloc[0][["slide_id", "sample_id"]]
        return self.get(slide_id=slide_id, sample_id=sample_id)
