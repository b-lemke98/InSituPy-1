# colorbar_module.py
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

# Create the static canvas for the colorbar
static_canvas = FigureCanvas(Figure(figsize=(5, 5)))

def update_colorbar(layer):
    import numpy as np
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec

    static_canvas.figure.clear()  # Clear the current figure
    gs = GridSpec(1, 1, top=1.2, bottom=0.6, left=-0.5, right=1.5)  # Define the grid spec
    axes = static_canvas.figure.add_subplot(gs[0])  # Add subplot with the grid spec

    # Get the colormap and normalization from the layer
    cmap = layer.colormap.name if hasattr(layer, 'colormap') else 'viridis'
    norm = Normalize(vmin=layer.data.min(), vmax=layer.data.max())

    colorbar = static_canvas.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation='horizontal')
    colorbar.set_label('Intensity')
    colorbar.ax.tick_params(labelsize=10)  # Adjust tick label size
    colorbar.set_ticks(np.linspace(norm.vmin, norm.vmax, num=5))  # Set the number of ticks
    axes.set_axis_off()
    static_canvas.draw()  # Redraw the canvas
