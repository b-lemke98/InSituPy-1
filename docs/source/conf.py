# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'InSituPy'
copyright = '2025, Johannes Wirth'
author = 'Johannes Wirth'

version = "0.6.6" #<<COOKIETEMPLE_FORCE_BUMP>>

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'myst_nb',
    'nbsphinx'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

nbsphinx_thumbnails = {
    'tutorials/01_InSituPy_demo_register_images': 'tutorials/demo_screenshots/common_features.png',
    'tutorials/07_InSituPy_InSituExperiment': 'tutorials/demo_screenshots/insituexperiment_structure.png',
}