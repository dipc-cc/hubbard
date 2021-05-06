# -*- coding: utf-8 -*-
#
# hubbard documentation build configuration file, created by
# sphinx-quickstart on Mon Oct 22 22:30:07 2018.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.mathjax',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.viewcode',
              'sphinx.ext.todo',
              'sphinx.ext.doctest',
              'sphinx.ext.napoleon',
              'sphinx.ext.coverage',
              'sphinx.ext.extlinks',
              'sphinx.ext.intersphinx',
              #'numpydoc',
              #'sphinx.ext.inheritance_diagram',
              'sphinx.ext.githubpages']

napoleon_numpy_docstring = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
#source_suffix = '.rst'
source_suffix = ['.rst']

# The master toctree document.
master_doc = 'index'
# prepend/append this snippet in _all_ sources
rst_prolog = """
.. highlight:: python
"""

import glob
from datetime import date
autosummary_generate = glob.glob('*.rst') + glob.glob('*/*.rst')
autosummary_generate = [f for f in autosummary_generate if 'api-generated' not in f]

# General information about the project.
project = u'hubbard'
copyright = f'2018-{date.today().year}, Sofia Sanz, Nick R. Papior, Mads Brandbyge and Thomas Frederiksen'
author = u'Sofia Sanz, Nick R. Papior, Mads Brandbyge and Thomas Frederiksen'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
#version = u''
# The full version, including alpha/beta/rc tags.
#release = u''

import hubbard.info as info
release = info.release
version = info.release

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Add __init__ classes to the documentation
autoclass_content = 'class'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'inherited-members': True,
    'show-inheritance': True,
}


# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = 'autolink'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
modindex_common_prefix = ['hubbard.']

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "hubbard documentation"

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "hubbard"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}

# If false, no index is generated.
html_use_modindex = True
html_use_index = True


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'hubbard'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'hubbard.tex', u'hubbard Documentation',
     u'Sofia Sanz, Nick R. Papior, Mads Brandbyge and Thomas Frederiksen', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'hubbard', u'hubbard Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'hubbard', u'hubbard Documentation',
     author, 'hubbard', 'Python package for mean-field Hubbard model simulations.',
     'Miscellaneous'),
]

# These two options should solve the "toctree contains reference to nonexisting document"
# problem.
# See here: numpydoc #69
class_members_toctree = False
# If this is false we do not have double method sections
numpydoc_show_class_members = False


# -----------------------------------------------------------------------------
# Intersphinx configuration
# -----------------------------------------------------------------------------
# Python, numpy, scipy and matplotlib specify https as the default objects.inv
# directory. So please retain these links.
intersphinx_mapping = {
    'python': ('https://docs.python.org/dev', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org', None),
    'sisl': ('http://zerothi.github.io/sisl/docs/latest', None),
}


def setup(app):
    import os
    import subprocess as sp
    if os.path.isfile('../conf_prepare.sh'):
        print("# Running ../conf_prepare.sh")
        sp.call(['bash', '../conf_prepale.sh'])
        print("\n# Done running ../run_prepare.sh")
    elif os.path.isfile('conf_prepare.sh'):
        print("# Running conf_prepare.sh")
        sp.call(['bash', 'conf_prepare.sh'])
        print("\n# Done running conf_prepare.sh")
    print("")
