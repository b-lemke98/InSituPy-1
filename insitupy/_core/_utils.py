
from insitupy._core.dataclasses import MultiCellData


def _get_cell_layer(
    cells: MultiCellData,
    cells_layer: str
):
    if cells_layer is None:
        layer = cells[cells.main_key]
    else:
        all_keys = cells.get_all_keys()
        if cells_layer not in all_keys:
            raise ValueError(f"cells_layer {cells_layer} not in layers: {all_keys}")
        layer = cells[cells_layer]

    return layer