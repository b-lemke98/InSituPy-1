# Sample-level analysis

These tutorials focus on the preprocessing, analysis and handling of individual samples of single-cell spatial transcriptomics data. These notebooks can be run from top to bottom and require the notebook about [automated image registration](../preprocessing/01_InSituPy_demo_register_images.ipynb) to be run first.

```{eval-rst}
.. card:: First analysis steps
    :link: 02_InSituPy_demo_analyze
    :link-type: doc

    Tutorial with introduction into first steps of analysis including filtering, preprocessing and dimensionality reduction.

.. card:: Annotations in `InSituPy`
    :link: 03_InSituPy_demo_annotations
    :link-type: doc

    Tutorial introducing how to import annotations and regions from external sources like QuPath or add them in the napari viewer.

.. card:: Crop data
    :link: 04_InSituPy_demo_crop
    :link-type: doc

    Tutorial showing how to crop data.

.. card:: Cell type annotation
    :link: 05_InSituPy_cell_type_annotation
    :link-type: doc

    Tutorial showing different options to perform cell type annotation.

.. card:: Explore gene expression along an axis
    :link: 06_InSituPy_gene_expression_along_axis
    :link-type: doc

    Demonstration on how to explore gene expression along an axis of the dataset.

.. card:: Build an `InSituData` object from scratch
    :link: 09_InSituPy_build_objects_from_scratch
    :link-type: doc

    Tutorial on how to generate an `InSituData` object from scratch.

```

.. card:: Perform segmentation with Proseg and add the results to `InSituData`
    :link: 11_InSituPy_add_proseg_data
    :link-type: doc

    Tutorial on how to improve cell segmentation with Proseg and add the data to `InSituData` for downstream analysis.

```

```{toctree}
:hidden: true
:maxdepth: 1

02_InSituPy_demo_analyze.ipynb
03_InSituPy_demo_annotations.ipynb
04_InSituPy_demo_crop.ipynb
05_InSituPy_cell_type_annotation.ipynb
06_InSituPy_gene_expression_along_axis.ipynb
09_InSituPy_build_objects_from_scratch.ipynb
```