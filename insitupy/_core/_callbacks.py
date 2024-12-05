import math

from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import insitupy._core.config as config
from insitupy import WITH_NAPARI
from insitupy._constants import POINTS_SYMBOL

if WITH_NAPARI:
    import napari


# geometry widget
def _update_keys_based_on_geom_type(widget, xdata):
    # retrieve current value
    current_geom_type = widget.geom_type.value
    current_key = widget.key.value

    # get either regions or annotations object
    geom_data = getattr(xdata, current_geom_type.lower())
    widget.key.choices = sorted(geom_data.metadata.keys(), key=str.casefold)

def _update_classes_on_key_change(widget, xdata):
    # get current values for geom_type and key
    current_geom_type = widget.geom_type.value
    current_key = widget.key.value

    # get either regions or annotations object
    geom_data = getattr(xdata, current_geom_type.lower())

    # update annot_class choices
    widget.annot_class.choices = ["all"] + sorted(geom_data.metadata[current_key]['classes'])

def _set_show_names_based_on_geom_type(widget):
    # retrieve current value
    current_geom_type = widget.geom_type.value

    # set the show_names tick box
    if current_geom_type == "Annotations":
        widget.show_names.value = False

    if current_geom_type == "Regions":
        widget.show_names.value = True


# Function to update the legend
def _update_categorical_legend(static_canvas, mapping, label, max_rows: int = 6):

    # Calculate the number of columns needed
    num_items = len(mapping)
    ncols = math.ceil(num_items / max_rows)

    static_canvas.figure.clear()  # Clear the current figure
    axes = static_canvas.figure.subplots()  # Create new axes
    legend_handles = [Line2D([0], [0],
                             marker='o', color='w', label=n,
                             markerfacecolor=c, markeredgecolor='k',
                             markersize=7) for n,c in mapping.items()]
    axes.legend(handles=legend_handles, loc="center", title=label, ncols=ncols,
                fontsize=8, title_fontsize=10,
                labelspacing=0.7, borderpad=0.5)
    axes.set_axis_off()
    static_canvas.draw()  # Redraw the canvas

def _update_continuous_legend(static_canvas, mapping, label):
    static_canvas.figure.clear()  # Clear the current figure
    gs = GridSpec(1, 1, top=1.2, bottom=0.6, left=-0.5, right=1.5)  # Define the grid spec
    axes = static_canvas.figure.add_subplot(gs[0])  # Add subplot with the grid spec

    colorbar = static_canvas.figure.colorbar(mapping, ax=axes, orientation='horizontal')
    colorbar.set_label(label, fontsize=10)
    colorbar.ax.tick_params(labelsize=8)  # Adjust tick label size
    #colorbar.set_ticks(np.linspace(norm.vmin, norm.vmax, num=5))  # Set the number of ticks
    axes.set_axis_off()
    static_canvas.draw()  # Redraw the canvas

def _update_colorlegend():

    #layer = viewer.layers[-1]

    # automatically get layer
    candidate_layers = [l for l in config.viewer.layers if l.name.startswith(f"{config.current_data_name}")]
    try:
        # always choose the candidate layer that is on top
        layer_name = candidate_layers[-1].name
    except IndexError:
        raise ValueError("No layer with cellular transcriptomic data found. First add a layer using the 'Show Data' widget.")

    # extract layer
    layer = config.viewer.layers[layer_name]

    # get values
    values = layer.properties["value"]
    color_values = layer.face_color

    from pandas.api.types import is_numeric_dtype

    from insitupy.plotting._colors import continuous_data_to_rgba
    if is_numeric_dtype(values):
        rgba_list, mapping = continuous_data_to_rgba(data=values,
                                cmap=layer.face_colormap.name,
                                #upper_climit_pct=upper_climit_pct,
                                return_mapping=True
                                )

        _update_continuous_legend(static_canvas=config.static_canvas,
                                  mapping=mapping,
                                  label=layer.name)


    else:
        # assume the data is categorical
        #mapping = {category: tuple(rgba) for category, rgba in zip(values, color_values)}
        unique_values = list(set(values))
        mapping = {v: tuple(color_values[list(values).index(v)]) for v in unique_values}
        # sort mapping dict
        mapping = {elem: mapping[elem] for elem in sorted(mapping.keys())}

        _update_categorical_legend(static_canvas=config.static_canvas,
                                   mapping=mapping, label=layer.name)

    # # create color mapping
    # rgba_list, mapping = _data_to_rgba(values, return_mapping=True)

    # if isinstance(mapping, dict):
    #     _update_categorical_legend(static_canvas=config.static_canvas,
    #                                mapping=mapping, label=layer.name)
    # else:
    #     _update_continuous_legend(static_canvas=config.static_canvas,
    #                               mapping=mapping,
    #                               label=layer.name)


def _refresh_widgets_after_data_change(xdata, points_widget, boundaries_widget):
    config.init_viewer_config(xdata)

    # set choices
    boundaries_widget.key.choices = config.masks

    # reset the currently selected key to None
    points_widget.value.value = None

    # add last addition to recent
    points_widget.recent.choices = sorted(config.recent_selections)
    points_widget.recent.value = None

    # set only the last cell layer visible
    cell_layers = []
    for elem in xdata.viewer.layers:
        if isinstance(elem, napari.layers.points.points.Points):
            if not elem.name.startswith(POINTS_SYMBOL):
                # only if the layer is not a point annotation layer, it is added
                cell_layers.append(elem)
    #point_layers = [elem for elem in xdata.viewer.layers if isinstance(elem, napari.layers.points.points.Points)]
    n_cell_layers = len(cell_layers)

    for i, l in enumerate(cell_layers):
        if i < n_cell_layers-1:
            l.visible = False