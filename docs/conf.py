# Sphinx documentation build configuration file
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'czbi_da_parez'
copyright = '2025, Martin Schätz'
author = 'Martin Schätz'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'alabaster'
