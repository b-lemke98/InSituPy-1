__author__ = "Johannes Wirth"
__email__ = "j.wirth@tum.de"
__version__ = "1.4.0"

# check if napari is available
try:
    import napari
    WITH_NAPARI = True
except ImportError:
    WITH_NAPARI = False

from ._core._deprecated import XeniumData
from . import image as im
from . import utils
from ._core.dataclasses import AnnotationsData, BoundariesData, ImageData
from ._core.insitudata import InSituData, read_xenium
from ._core.io import read_celldata

__all__ = [
    "InSituData",
    "AnnotationsData",
    "BoundariesData",
    "ImageData",
    "im",
    "utils"
]