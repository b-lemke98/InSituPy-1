import os
from copy import deepcopy
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure

import insitupy
from insitupy._constants import LOAD_FUNCS
from insitupy._exceptions import ModalityNotFoundError
from insitupy.io.files import check_overwrite_and_remove_if_true
from insitupy.utils.utils import convert_to_list
from insitupy.utils.utils import textformat as tf

from ..io.plots import save_and_show_figure
from ..utils.utils import get_nrows_maxcols


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

    def iterdata(self):
        """Iterate over the metadata rows and corresponding data.

        Yields:
            tuple: A tuple containing the index, metadata row as a Series, and the corresponding data.
        """
        for idx, row in self._metadata.iterrows():
            yield row, self._data[idx]

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

    def plot_umaps(self,
                   color: Optional[str] = None,
                   title_columns: Optional[Union[List[str], str]] = None,
                   title_size: int = 20,
                   max_cols: int = 4,
                   figsize: Tuple[int, int] = (8,6),
                   savepath: Optional[os.PathLike] = None,
                   save_only: bool = False,
                   show: bool = True,
                   fig: Optional[Figure] = None,
                   dpi_save: int = 300,
                   **kwargs):
        """Create a plot with UMAPs of all datasets as subplots using scanpy's pl.umap function.

        Args:
            color (str or list of str, optional): Keys for annotations of observations/cells or variables/genes to color the plot. Defaults to None.
            title_columns (str or list of str, optional): List of column names from metadata to use for subplot titles. Defaults to None.
            max_cols (int, optional): Maximum number of columns for subplots. Defaults to 4.
            **kwargs: Additional keyword arguments to pass to sc.pl.umap.
            figsize (tuple, optional): Figure size. Defaults to (8, 6).
            savepath (optional): Path to save the plot.
            save_only (bool, optional): Whether to only save the plot without showing. Defaults to False.
            show (bool, optional): Whether to show the plot. Defaults to True.
            fig (optional): Figure to plot on.
            dpi_save (int, optional): DPI for saving the plot. Defaults to 300.
        """
        num_datasets = len(self._data)
        n_plots, n_rows, max_cols = get_nrows_maxcols(self._data, max_cols)
        fig, axes = plt.subplots(n_rows, max_cols, figsize=(figsize[0]*max_cols, figsize[1]*n_rows))

        # make sure title_columns is a list
        if title_columns is not None:
            title_columns = convert_to_list(title_columns)

        for idx, (metadata_row, dataset) in enumerate(self.iterdata()):
            ax = axes[idx] if num_datasets > 1 else axes
            # Assuming each dataset has an AnnData object or can be converted to one
            adata = dataset.cells.matrix
            sc.pl.umap(adata, ax=ax, color=color, show=False, **kwargs)

            if title_columns:
                title = " - ".join(str(metadata_row[col]) for col in title_columns if col in metadata_row)
                ax.set_title(title, fontdict={"fontsize": title_size})
            else:
                ax.set_title(f"Dataset {idx + 1}", fontdict={"fontsize": title_size})

        if n_plots > 1:

            # check if there are empty plots remaining
            i = n_plots
            while i < n_rows * max_cols:
                i+=1
                # remove empty plots
                axes[i].set_axis_off()
        if show:
            #fig.tight_layout()
            save_and_show_figure(savepath=savepath, fig=fig, save_only=save_only, dpi_save=dpi_save, tight=True)
        else:
            return fig, axes

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
        dataset_paths = sorted([elem for elem in path.glob("data-*") if elem.is_dir()])
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

    @classmethod
    def from_config(cls, config_path: Union[str, os.PathLike, Path]):
        """Create an InSituExperiment object from a configuration file.

        Args:
            config_path (Union[str, os.PathLike, Path]): The path to the configuration CSV or Excel file.

        Returns:
            InSituExperiment: A new InSituExperiment object with the loaded data and metadata.
        """
        config_path = Path(config_path)

        # Determine file type and read the configuration file
        if config_path.suffix in ['.csv']:
            config = pd.read_csv(config_path)
        elif config_path.suffix in ['.xlsx', '.xls']:
            config = pd.read_excel(config_path)
        else:
            raise ValueError("Unsupported file type. Please provide a CSV or Excel file.")

        # Ensure the 'directory' column exists
        if 'directory' not in config.columns:
            raise ValueError("The configuration file must contain a 'directory' column.")

        # Initialize a new InSituExperiment object
        experiment = cls()

        # Iterate over each row in the configuration file
        for _, row in config.iterrows():
            dataset_path = Path(row['directory'])
            dataset = insitupy.read_xenium(dataset_path)
            experiment._data.append(dataset)

            # Extract metadata from the row, excluding the 'directory' column
            metadata = row.drop(labels=['directory']).to_dict()
            metadata['uid'] = str(uuid4()).split("-")[0]
            metadata['slide_id'] = dataset.slide_id
            metadata['sample_id'] = dataset.sample_id

            # Append the metadata to the experiment's metadata DataFrame
            experiment._metadata = pd.concat([experiment._metadata, pd.DataFrame([metadata])], ignore_index=True)

        return experiment



    def remove_history(self):
        for xd in self._data:
            print(xd.sample_id)
            xd.remove_history()

    def save(self):
        if self.path is None:
            print("No save path found in '.self'. First save the InSituExperiment using '.saveas()'.")
            return
        else:
            parent_path_identical = [d.path.parent == self.path for d in self.data]
            if not np.all(parent_path_identical):
                print(f"Saving process failed. Save path of some InSituData objects did not lie inside the InSituExperiment save path: {self.metadata['uid'][parent_path_identical].values}")
            else:
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
            subfolder_path = path / f"data-{str(index).zfill(3)}"
            dataset.saveas(subfolder_path, verbose=False)

        # Optionally, save the metadata as a CSV file
        metadata_path = os.path.join(path, "metadata.csv")
        self._metadata.to_csv(metadata_path, index=True)

        print("Saved.") if verbose else None

    def show(self, index: int, return_viewer: bool = True):
        """
        Displays the dataset at the specified index.

        Args:
            index (int): The index of the dataset to display.
            return_viewer (bool, optional): If True, returns the viewer object of the dataset. Defaults to True.

        Returns:
            Viewer: The viewer object of the dataset if return_viewer is True.
        """
        dataset = self.data[index]
        dataset.show()
        if return_viewer:
            return dataset.viewer

