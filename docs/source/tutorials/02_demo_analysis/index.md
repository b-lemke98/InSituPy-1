# Sample-level analysis

These tutorials focus on the preprocessing, analysis and handling of individual samples of single-cell spatial transcriptomics data. These notebooks can be run from top to bottom and require the notebook about [automated image registration](../preprocessing/01_InSituPy_demo_register_images.ipynb) to be run first.

```{eval-rst}
.. card:: 01: Automated image registration
    :link: 01_InSituPy_demo_register_images
    :link-type: doc
    :link-alt: Tutorial on how to use the automated image registration pipeline in `InSituPy`.

    Tutorial showing how to use the automated image registration pipeline to register histological stainings or IF stainings performed on the same slide as the scST assay.


.. card:: 02: First analysis steps
    :link: 02_InSituPy_demo_analyze
    :link-type: doc

    Tutorial with introduction into first steps of analysis including filtering, preprocessing and dimensionality reduction.

.. card:: 03: Annotations in `InSituPy`
    :link: 03_InSituPy_demo_annotations
    :link-type: doc

    Tutorial introducing how to import annotations and regions from external sources like QuPath or add them in the napari viewer.

.. card:: 04: Crop data
    :link: 04_InSituPy_demo_crop
    :link-type: doc

    Tutorial showing how to crop data.

.. card:: 05: Cell type annotation
    :link: 05_InSituPy_cell_type_annotation
    :link-type: doc

    Tutorial showing different options to perform cell type annotation.

.. card:: 06: Explore gene expression along an axis
    :link: 06_InSituPy_gene_expression_along_axis
    :link-type: doc

    Demonstration on how to explore gene expression along an axis of the dataset.
```

```{toctree}
:hidden: false
:maxdepth: 1
:glob:

*
```