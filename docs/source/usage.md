## Installation

### Prerequisites

**Create and activate a conda environment:**

   ```bash
   conda create --name insitupy python=3.9
   conda activate insitupy
   ```

### Method 1: From PyPi

   ```bash
   pip install insitupy-spatial
   ```

### Method 2: Installation from Cloned Repository

1. **Clone the repository to your local machine:**

   ```bash
   git clone https://github.com/SpatialPathology/InSituPy.git
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

### Method 3: Direct Installation from GitHub

1. **Install directly from GitHub:**

   ```bash
   # for installation without napari use
   pip install git+https://github.com/SpatialPathology/InSituPy.git
   ```

Make sure you have Conda installed on your system before proceeding with these steps. If not, you can install Miniconda or Anaconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).

To ensure that the InSituPy package is available as a kernel in Jupyter notebooks within your conda environment, you can follow the instructions [here](https://ipython.readthedocs.io/en/stable/install/kernel_install.html).

## Getting started

### Documentation

__The Documentation is not online yet and will be made public in the next weeks__!

(For detailed instructions on using InSituPy, refer to the [official documentation](https://InSituPy.readthedocs.io), which will be made public after publication. The documentation will provide comprehensive guides on installation, usage, and advanced features.)


### Tutorials

Explore the tutorials in `./notebooks/` to learn how to use InSituPy:

#### Sample level analysis

These tutorials focus on the preprocessing, analysis and handling of individual samples.

1. [Registration of additional images](notebooks/01_InSituPy_demo_register_images.ipynb) - Learn how to register additional images to the spatial transcriptomics data.
    1. Alternatively this is also implemented for an example dataset from [pancreatic cancer](notebooks/pancreas/01panc_InSituPy_demo_register_images.ipynb)
3. [Basic analysis functionalities](notebooks/02_InSituPy_demo_analyze.ipynb) - Learn about the basic functionalities, such as loading of data, basic preprocessing and interactive visualization with napari.
4. [Add annotations](notebooks/03_InSituPy_demo_annotations.ipynb) - Learn how to add annotations from external software such as [QuPath](https://qupath.github.io/) and do annotations in the napari viewer.
5. [Crop data](notebooks/04_InSituPy_demo_crop.ipynb) - Learn how to crop your data to focus your analysis on specific areas in the tissue.
6. [Cell type annotation](notebooks/05_InSituPy_cell_type_annotation.ipynb) - Shows an example workflow to annotate the cell types.
7. [Explore gene expression along axis](notebooks/06_InSituPy_gene_expression_along_axis_pattern.ipynb) - Example cases showing how to correlate gene expression with e.g. the distance to histological annotations.
8. [Build an `InSituData` object from scratch](notebooks/09_InSituPy_build_objects_from_scratch.ipynb) - General introduction on how to build an `InSituData` object from scratch.

#### Experiment-level analysis

This set of tutorials focuses on

1. [Analyze multiple samples at once with InSituPy](notebooks/07_InSituPy_InSituExperiment.ipynb) - Introduces the main concepts behind the `InSituExperiment` class and how to work with multiple samples at once.
2. [Differential gene expression analysis](notebooks/08_InSituPy_differential_gene_expression.ipynb) - Perform differential gene expression analysis within one sample and across multiple samples.

### Example data

If you want to test the pipeline on different example datasets, [this notebook](notebooks/00_InSituPy_demo_datasets.ipynb) provides an overview of functions to download _Xenium In Situ_ data from official sources.
