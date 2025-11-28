# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = "Gloria"
copyright = "2025, Benjamin Kambs, Patrik Wollgarten"
author = "Benjamin Kambs, Patrik Wollgarten"
release = "0.1.3"
# Gloria
import gloria

version = gloria.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_inline_tabs",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
    "sphinx_design",
]


# Show type hints only in the parameter descriptions (not in the signature)
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "sktime": ("https://www.sktime.net/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

templates_path = ["_templates"]
exclude_patterns = []

# Use field lists instead of NumPy/Google table formatting
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True
napoleon_use_admonition_for_examples = False
# napoleon_use_keyword = True  # Treat kwargs like other parameters


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_favicon = "_static/gloria_favicon.ico"

html_theme_options = {
    "logo": {
        "image_light": "_static/glorialogo.png",
        "image_dark": "_static/glorialogo_white.png",
    },
    "navigation_with_keys": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/e-dyn/gloria",
            "icon": "fab fa-github",
        },
    ],
    "use_edit_page_button": False,
    "show_prev_next": True,
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "primary_sidebar_end": ["sidebar-ethical-ads"],
    "footer_start": [],
    "footer_end": [],
}

html_show_sourcelink = False  # sicherheitshalber trotzdem setzen
html_copy_source = False  # optional, keine .txt-Dateien erzeugen

html_sidebars = {
    "**": ["sidebar-nav-bs.html"],
}
