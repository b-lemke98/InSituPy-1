from warnings import warn


### DEPRECATED FUNCTIONS
def read_all(self, *args, **kwargs):
    warn("`read_all` is deprecated. Use `load_all` instead.", DeprecationWarning, stacklevel=2)

def read_annotations(self, *args, **kwargs):
    warn("`read_annotations` is deprecated. Use `load_annotations` instead.", DeprecationWarning, stacklevel=2)

def read_regions(self, *args, **kwargs):
    warn("`read_regions` is deprecated. Use `load_regions` instead.", DeprecationWarning, stacklevel=2)

def read_cells(self, *args, **kwargs):
    warn("`read_cells` is deprecated. Use `load_cells` instead.", DeprecationWarning, stacklevel=2)

def read_images(self, *args, **kwargs):
    warn("`read_images` is deprecated. Use `load_images` instead.", DeprecationWarning, stacklevel=2)

def read_transcripts(self, *args, **kwargs):
    warn("`read_transcripts` is deprecated. Use `load_transcripts` instead.", DeprecationWarning, stacklevel=2)

def read_xenium(self, *args, **kwargs):
    warn("`read_xenium` is deprecated. Use `read(mode='xenium')` instead.", DeprecationWarning, stacklevel=2)