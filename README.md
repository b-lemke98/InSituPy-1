# InSituPy: Python Package for the Analysis of _Xenium In Situ_ Data

InSituPy is a Python package designed to facilitate the analysis of in situ sequencing data generated with the [_Xenium In Situ_](https://www.10xgenomics.com/platforms/xenium) methodology. With InSituPy, you can easily load, visualize, and analyze the data, enabling and simplifying the comprehensive exploration of spatial gene expression patterns within tissues.

## Installation

To install InSituPy within a conda environment, you can follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/jwrth/InSituPy.git
   ```

2. Navigate to the cloned repository:

   ```bash
   cd insitupy
   ```

3. Create and activate a conda environment:

   ```bash
   conda create --name insitupy python=3.9
   conda activate insitupy
   ```

4. Install the package using `pip` within the conda environment:

   ```bash
   pip install .
   ```

Make sure you have Conda installed on your system before proceeding with these steps. If not, you can install Miniconda or Anaconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).

## Getting Started

For detailed instructions on using InSituPy, refer to the [official documentation](https://InSituPy.readthedocs.io). The documentation provides comprehensive guides on installation, usage, and advanced features.

## Tutorials

Explore the tutorials to learn how to use InSituPy effectively:

1. [Download example data for tutorial](notebooks/00_InSituPy_demo_download_data.ipynb) - Download _Xenium In Situ_ example data for the subsequent tutorials.
2. [Basic functionalities](notebooks/00_InSituPy_demo_download_data.ipynb) - Learn about the basic functionalities, such as loading of data, image registration and interactive visualization with napari.

## Features

- **Data Preprocessing:** InSituPy provides functions for normalizing, filtering, and transforming raw in situ transcriptomics data.

- **Interactive Visualization:** Create interactive plots using [napari](https://napari.org/stable/#) to easily explore spatial gene expression patterns.

- **Annotation:** Annotate _Xenium In Situ_ data in the napari viewer or import annotations from external tools like [QuPath](https://qupath.github.io/).

## Contributing

Contributions are welcome! If you find any issues or have suggestions for new features, please open an [issue](https://github.com/jwrth/InSituPy/issues) or submit a pull request.

## License

InSituPy is licensed under the [MIT License](LICENSE).

---

**InSituPy** is developed and maintained by [jwrth](https://github.com/jwrth). Feedback is highly appreciated and hopefully **InSituPy** helps you with your _Xenium In Situ_ analysis. The package is thought to be a starting point to simplify the analysis of in situ sequencing data in Python and it would be exciting to integrate functionalities into larger and more comprehensive data structures.
