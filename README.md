# InSituPy: A framework for histology-guided, multi-sample  analysis of single-cell spatial transcriptomics data

<p align="center">
   <img src="logo/insitupy_logo.png" width="500">
</p>

InSituPy is a Python package designed to facilitate the analysis of single-cell spatial transcriptomics data. With InSituPy, you can easily load, visualize, and analyze the data, enabling and simplifying the comprehensive exploration of spatial gene expression patterns within tissue sections and across multiple samples.
Currently the analysis is focused on data from the [_Xenium In Situ_](https://www.10xgenomics.com/platforms/xenium) methodology but a broader range of reading functions will be implemented in the future.

## Latest changes

## Update to `0.5.0`
### Major changes in reading/loading logic!
This might conflict with the backwards compatibility of this version! If there are issues with loading reading `InSituPy` projects saved with older version, please let me know to find workarounds!
* Reduced focus on Xenium method in data structure
* `InSituData.read()` substitutes `read_xenium` for reading of `InSituPy` projects. `read_xenium` used now to read data from Xenium data folders

## Installation

### Prerequisites

1. **Create and activate a conda environment:**

   ```bash
   conda create --name insitupy python=3.9
   conda activate insitupy
   ```

### Method 1: Installation from Cloned Repository

1. **Clone the repository to your local machine:**

   ```bash
   git clone https://github.com/jwrth/InSituPy.git
   ```

2. **Navigate to the cloned repository and select the right branch:**

   ```bash
   cd InSituPy

   # Optionally: switch to dev branch
   git checkout dev
   ```

3. **Install the required packages using `pip` within the conda environment:**

   ```bash
   # basic installation
   pip install .

   # for developmental purposes add the -e flag
   pip install -e .
   ```

### Method 2: Direct Installation from GitHub

1. **Install directly from GitHub:**

   ```bash
   # for installation without napari use
   pip install git+https://github.com/jwrth/InSituPy.git
   ```

Make sure you have Conda installed on your system before proceeding with these steps. If not, you can install Miniconda or Anaconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).

To ensure that the InSituPy package is available as a kernel in Jupyter notebooks within your conda environment, you can follow the instructions [here](https://ipython.readthedocs.io/en/stable/install/kernel_install.html).

## Getting started

### Documentation

For detailed instructions on using InSituPy, refer to the [official documentation](https://InSituPy.readthedocs.io), which will be made public after publication. The documentation will provide comprehensive guides on installation, usage, and advanced features.


### Tutorials

Explore the tutorials in `./notebooks/` to learn how to use InSituPy:

#### Sample level analysis

These tutorials focus on the preprocessing, analysis and handling of individual samples.

1. [Registration of additional images](notebooks/01_InSituPy_demo_register_images.ipynb) - Learn how to register additional images to the spatial transcriptomics data.
2. [Basic analysis functionalities](notebooks/02_InSituPy_demo_analyze.ipynb) - Learn about the basic functionalities, such as loading of data, basic preprocessing and interactive visualization with napari.
3. [Add annotations](notebooks/03_InSituPy_demo_annotations.ipynb) - Learn how to add annotations from external software such as [QuPath](https://qupath.github.io/) and do annotations in the napari viewer.
4. [Crop data](notebooks/04_InSituPy_demo_crop.ipynb) - Learn how to crop your data to focus your analysis on specific areas in the tissue.
5. [Cell type annotation](notebooks/05_InSituPy_cell_type_annotation.ipynb) - Shows an example workflow to annotate the cell types.
6. [Explore gene expression along axis](notebooks/06_InSituPy_gene_expression_along_axis_pattern.ipynb) - Example cases showing how to correlate gene expression with e.g. the distance to histological annotations.
7. [Build an `InSituData` object from scratch](notebooks/09_InSituPy_build_objects_from_scratch.ipynb) - General introduction on how to build an `InSituData` object from scratch.

#### Experiment-level analysis

This set of tutorials focuses on

1. [Analyze multiple samples at once with InSituPy](notebooks/07_InSituPy_InSituExperiment.ipynb) - Introduces the main concepts behind the `InSituExperiment` class and how to work with multiple samples at once.
2. [Differential gene expression analysis](notebooks/08_InSituPy_differential_gene_expression.ipynb) - Perform differential gene expression analysis within one sample and across multiple samples.

### Example data

If you want to test the pipeline on different example datasets, [this notebook](notebooks/00_InSituPy_demo_datasets.ipynb) provides an overview of functions to download _Xenium In Situ_ data from official sources.

## Features

- **Data Preprocessing:** InSituPy provides functions for normalizing, filtering, and transforming raw in situ transcriptomics data.
- **Interactive Visualization:** Create interactive plots using [napari](https://napari.org/stable/#) to easily explore spatial gene expression patterns.
- **Annotation:** Annotate _Xenium In Situ_ data in the napari viewer or import annotations from external tools like [QuPath](https://qupath.github.io/).
- **Multi-sample analysis:** Perform analysis on an experiment-level, i.e. with multiple samples at once.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for new features, please open an [issue](https://github.com/SpatialPathology/InSituPy/issues) or submit a pull request.

## License

InSituPy is licensed under the [BSD-3-Clause](LICENSE).

---

**InSituPy** is developed and maintained by [Johannes Wirth](https://github.com/jwrth) and [Anna Chernysheva](https://github.com/annachernysheva179). Feedback is highly appreciated and hopefully **InSituPy** helps you with your analysis of spatial transcriptomics data. The package is thought to be a starting point to simplify the analysis of in situ sequencing data in Python and it would be exciting to integrate functionalities into larger and more comprehensive data structures. Currently, the framework focuses on the analysis of _Xenium In Situ_ data but it is planned to integrate more methodologies and any support on this is highly welcomed.
