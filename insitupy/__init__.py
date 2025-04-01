__author__ = "Johannes Wirth"
__email__ = "j.wirth@tum.de"
__version__ = "0.7.0"

# check if napari is available
try:
    import napari
    WITH_NAPARI = True
except ImportError:
    WITH_NAPARI = False

from insitupy._constants import CACHE

from . import images as im
from . import io
from . import plotting as pl
from . import preprocessing as pp
from . import tools as tl
from . import utils
from ._core.dataclasses import AnnotationsData, BoundariesData, ImageData
from ._core.insitudata import InSituData
from ._core.insituexperiment import InSituExperiment
from .images.registration import register_images
from .io.readers import read_xenium
from .palettes import CustomPalettes
from .tools.dge import differential_gene_expression
from .tools.distance import calc_distance_of_cells_from

__all__ = [
    "InSituData",
    "AnnotationsData",
    "BoundariesData",
    "ImageData",
    "im",
    "utils"
]