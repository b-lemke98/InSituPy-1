# https://pyapp-kit.github.io/magicgui/decorators/
# from magicgui import magicgui
# from enum import Enum

# class Medium(Enum):
#     Glass = 1.520
#     Oil = 1.515
#     Water = 1.333
#     Air = 1.0003

# class Whoop(Enum):
#     blubb = 1.520
#     check = 1.515
#     foo = 1.333
#     test = 1.0003

# # decorate your function with the @magicgui decorator
# @magicgui(call_button="calculate", result_widget=True)
# def snells_law(aoi=30.0, n1=Medium.Glass, n2=Medium.Water, degrees=True):
#     import math

#     aoi = math.radians(aoi) if degrees else aoi
#     try:
#         result = math.asin(n1.value * math.sin(aoi) / n2.value)
#         return math.degrees(result) if degrees else result
#     except ValueError:
#         return "Total internal reflection!"

# # your function is now capable of showing a GUI
# #snells_law.show(run=True)

# snells_law.n1.value = Whoop.blubb
# snells_law.show(run=True)

# from magicgui import magicgui
# from enum import Enum, auto

# class Choices(Enum):
#     a = auto()
#     b = auto()
#     c = auto()
#     d = auto()


# @magicgui(x=dict(widget_type='Select', choices=Choices))
# def test(x):
#     print(x)

# test.show(run=True)

# from typing import Tuple
# import napari
# from napari.types import LayerDataTuple
# from magicgui import magicgui
# from pandas.api.types import is_numeric_dtype

# class PointWidgets:
#     def __init__(self):
#         self.genes = ["A", "B", "C"]
        
#     @magicgui(call_button='Add', observation={'choices': 'genes'})
#     def test(self):
#         print('hey')

# import napari
# import magicgui
# from typing import List

# def my_fancy_choices_function(gui) -> List[str]:
#     # `gui` will be the combobox instance... if you want, you can traverse
#     # the widget parent chain to find the viewer and do whatever you want.
#     # but here, we just return a simple list of strings that we want
#     # the combobox to show
#     return ["otsu", "opening"]

# @magicgui(layer_name={"choices": my_fancy_choices_function})
# def my_gui(layer_name: str, viewer: napari.Viewer):
#     # the current layer_name choice to get the viewer.
#     selected_layer = viewer.layers[layer_name]
#     # go nuts with your layer.

# my_gui.show(run=True)

# import napari
# from magicgui import magicgui
# from typing import List

# def my_fancy_choices_function(gui) -> List[str]:
#     # `gui` will be the combobox instance... if you want, you can traverse
#     # the widget parent chain to find the viewer and do whatever you want.
#     # but here, we just return a simple list of strings that we want
#     # the combobox to show
#     return ["otsu", "opening"]

# @magicgui(layer_name={"choices": my_fancy_choices_function})
# def my_gui(layer_name: str, viewer: napari.Viewer):
#     # the current layer_name choice to get the viewer.
#     selected_layer = viewer.layers[layer_name]
#     # go nuts with your layer.

# my_gui.show(run=True)


import napari
from magicgui import magicgui
from typing import List

class FancyChoices:
    def __init__(self):
        self.choices = ["otsu", "opening"]
        
    def _get_choices(self):
        return self.choices

    @magicgui(layer_name={"choices": _get_choices})
    def my_gui(layer_name: str, viewer: napari.Viewer):
        # the current layer_name choice to get the viewer.
        selected_layer = viewer.layers[layer_name]
        # go nuts with your layer.

fc = FancyChoices()
print(fc.choices)
fc.my_gui.show(run=True)

# from magicgui.experimental import guiclass, button
# from typing import List

# @guiclass
# class MyDataclass:
#     a: int = 0
#     b: str = 'hello'
#     c: tuple = tuple(list("abc"))

#     # @button
#     # def compute(self):
#     #     print(self.a, self.b, self.c)

# obj = MyDataclass(a=10, b='foo', c='a')
# obj.gui.show(run=True)