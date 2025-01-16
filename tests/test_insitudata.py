from pathlib import Path
from insitupy.datasets._datasets import xenium_test_dataset

BAYSOR_PATH = Path("tests/data/baysor_output-slide__region__20241212__134825__j1_b1_s1")
image_x = 3522
image_y = 5789
n_cells = 23
n_transripts = 1985
n_genes = 5006
n_images = 1
image_names = ["nuclei"]

def test_read():
    xd = xenium_test_dataset()
    xd.load_all()

    assert len(xd.images.metadata) == n_images
    assert xd.images["nuclei"][0].shape == (image_x, image_y)

    assert len(xd.cells.boundaries.metadata) == 2
    assert xd.cells.boundaries["cellular"].shape == (image_x, image_y)
    assert xd.cells.boundaries["nuclear"].shape == (image_x, image_y)

    assert xd.transcripts.shape == (n_transripts, 9)
    assert xd.cells.matrix.shape == (n_cells, n_genes)


def test_baysor():
    xd = xenium_test_dataset()
    xd.load_all()
    xd.add_baysor(BAYSOR_PATH)
    assert xd.alt is not None
    assert len(xd.alt) == 1
    assert "baysor" in xd.alt.keys()
    assert xd.alt["baysor"].matrix is not None
    assert xd.alt["baysor"].matrix.shape == (18, 11)
    assert xd.alt["baysor"].boundaries is not None
    assert xd.alt["baysor"].boundaries
    assert 'cellular' in xd.alt["baysor"].boundaries.metadata.keys()
    

def test_functions():
    xd = xenium_test_dataset()
    xd.load_all()
    xd.normalize_and_transform(transformation_method="sqrt")
    xd.reduce_dimensions(umap=True, tsne=False)
    for key in ['spatial', 'X_pca', 'X_umap']:
        assert key in xd.cells.matrix.obsm.keys()
    assert "leiden" in xd.cells.matrix.obs.columns