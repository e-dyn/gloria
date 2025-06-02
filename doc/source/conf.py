# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = "Gloria"
copyright = "2025, Benjamin Kambs, Patrik Wollgarten"
author = "Benjamin Kambs, Patrik Wollgarten"
release = "0.1.0.dev1"
# Gloria
import gloria

version = gloria.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx_inline_tabs"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "logo": {
        "image_light": "_static/glorialogo.png",
        "image_dark": "_static/glorialogo_dark.png",
    }
}
