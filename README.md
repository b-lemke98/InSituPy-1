# InSituPy: A framework for histology-guided, multi-sample  analysis of single-cell spatial transcriptomics data

<p align="center">
   <img src="./docs/source/_static/img/insitupy_logo_with_text.png" width="500">
</p>

InSituPy is a Python package designed to facilitate the analysis of single-cell spatial transcriptomics data. With InSituPy, you can easily load, visualize, and analyze the data, enabling and simplifying the comprehensive exploration of spatial gene expression patterns within tissue sections and across multiple samples.
Currently the analysis is focused on data from the [_Xenium In Situ_](https://www.10xgenomics.com/platforms/xenium) methodology but a broader range of reading functions will be implemented in the future.

## Latest changes

*!!!Warning: This repository is under very active development and it cannot be ruled out that changes might impair backwards compatibility. If you observe any such thing, please feel free to contact us to solve the problem. Thanks!*

For the latest developments check out the [releases](https://github.com/SpatialPathology/InSituPy/releases).

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

For detailed instructions on using InSituPy, refer to the [official documentation](https://InSituPy.readthedocs.io).

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

**InSituPy** is developed and maintained by [Johannes Wirth](https://github.com/jwrth) and [Anna Chernysheva](https://github.com/annachernysheva179). Feedback is highly appreciated and hopefully **InSituPy** helps you with your analysis of spatial transcriptomics data. The package is thought to be a starting point to simplify the analysis of in situ sequencing data in Python and it would be exciting to integrate functionalities for larger and more comprehensive data structures. Currently, the framework focuses on the analysis of _Xenium In Situ_ data but it is planned to integrate more methodologies and any support on this is highly welcomed.
