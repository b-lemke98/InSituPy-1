import napari
import numpy as np
from napari.layers import Image
from magicgui import magicgui

@magicgui(image={'label': 'Pick an Image'})
def my_widget(image: Image):
    raise ValueError("Error")
    print('hello')

viewer = napari.view_image(np.random.rand(64, 64), name="My Image")
viewer.window.add_dock_widget(my_widget)