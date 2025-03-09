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

autosummary_generate = True
autodoc_process_signature = True
autodoc_member_order = "groupwise"
default_role = "literal"
napoleon_google_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
myst_heading_anchors = 3  # create anchors for h1-h3
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
    ".md": "myst-nb"
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_book_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

nbsphinx_thumbnails = {
    'tutorials/01_InSituPy_demo_register_images': 'tutorials/demo_screenshots/common_features.png',
    'tutorials/07_InSituPy_InSituExperiment': 'tutorials/demo_screenshots/insituexperiment_structure.png',
}