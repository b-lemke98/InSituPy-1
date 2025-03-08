# Configuration file for the Sphinx documentation builder.

# -- Project information

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath('../..'))

def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


# -- Project information

project = "InSituPy"
author = "Johannes Wirth"
copyright = "{datetime.now():%Y}, {author}"
release = get_version("../../insitupy/__init__.py")
version = get_version("../../insitupy/__init__.py")

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'nbsphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_book_theme'
html_title = project
html_logo = "insitupy_logo.png"

# -- Options for EPUB output
epub_show_urls = 'footnote'

nbsphinx_thumbnails = {
    'tutorials/01_InSituPy_demo_register_images': 'tutorials/demo_screenshots/common_features.png',
    'tutorials/07_InSituPy_InSituExperiment': 'tutorials/demo_screenshots/insituexperiment_structure.png',
}