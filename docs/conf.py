import sys
import os

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'levin'
copyright = '2024, Robert Reischke'
author = 'Robert Reischke'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['numpydoc', 'sphinx.ext.autosectionlabel']
numpydoc_show_class_members = False
#autoclass_content = 'init'
sys.path.insert(0, os.path.abspath("./../levin_bessel"))
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
autodoc_default_options = {
    "members": True,
    "private-members": True
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_logo = 'levin_logo.jpeg'
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}

# autodoc_mock_imports = ["levin", "camb", "hmf"]


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []