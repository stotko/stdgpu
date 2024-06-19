# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import pathlib

project = "stdgpu"
version = "Latest"
release = version
copyright = "2019, Patrick Stotko"
# author = "Patrick Stotko"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinxcontrib.doxylink",
    "myst_parser",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

doxylink = {
    "stdgpu": (
        str(pathlib.Path(__file__).parent / "doxygen" / "tagfile.xml"),
        "doxygen",
    ),
}


# templates_path = ["_templates"]
# exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]


html_theme_options = {
    # Sidebar
    "logo": {
        "text": "Latest",
    },
    # Header
    "use_download_button": False,
    "use_repository_button": True,
    "use_fullscreen_button": False,
    "repository_url": "https://github.com/stotko/stdgpu",
    # Footer
    "extra_footer": 'Made with <a href="https://www.sphinx-doc.org/">Sphinx</a>, <a href="https://www.doxygen.org/index.html">Doxygen</a>, <a href="https://boschglobal.github.io/doxysphinx/">Doxysphinx</a> and <a href="https://sphinx-book-theme.readthedocs.io/">sphinx-book-theme</a>, <a href="https://jothepro.github.io/doxygen-awesome-css/">Doxygen Awesome</a>',
    # Code fragments
    "pygments_light_style": "a11y-high-contrast-light",  # same as in sphinx-book-theme
    "pygments_dark_style": "a11y-dark",
}

html_favicon = "_static/stdgpu_logo.ico"
html_logo = "_static/stdgpu_logo.png"

html_last_updated_fmt = "%Y-%m-%d"

html_css_files = [
    "stdgpu_custom_sphinx.css",
]


html_copy_source = False
