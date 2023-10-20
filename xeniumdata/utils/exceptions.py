class ModuleNotFoundOnWindows(ModuleNotFoundError):
    '''
    Code from https://github.com/theislab/scib/blob/main/scib/exceptions.py
    Information about structure: https://careerkarma.com/blog/python-super/
    '''

    def __init__(self, exception):
        self.message = f"\n{exception.name} is not installed. " \
                       "This package could be problematic to install on Windows."
        super().__init__(self.message)
        
class XeniumDataRepeatedCropError(Exception):
    """Exception raised if it is attempted to crop a
    XeniumData object multiple times with the same cropping window.

    Attributes:
        xlim: Limits on x-axis.
        ylim: Limits on y-axis.
    """

    def __init__(self, xlim, ylim):
        self.xlim = xlim
        self.ylim = ylim
        self.message = f"\nXeniumData object has been cropped with the same limits before:\n" \
            f"xlim -> {xlim}\n" \
            f"ylim -> {ylim}" 
        super().__init__(self.message)
        
class WrongNapariLayerTypeError(Exception):
    """Exception raised if current layer has not the right format.

    Attributes:
        found: Napari layer type found.
        wanted: Napari layer type wanted.
    """

    def __init__(self, found, wanted):
        self.message = f"\nNapari layer has wrong format ({found}) instead of {wanted}"
        super().__init__(self.message)

class NotOneElementError(Exception):
    """Exception raised if list contains not exactly one element.

    Attributes:
        list: List which does not contain one element.
    """

    def __init__(self, l):
        self.message = f"List was expected to contain one element but contained {len(l)}"
        super().__init__(self.message)