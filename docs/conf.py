"""Sphinx configuration for the OpenMC Plasma Source documentation."""

project = "OpenMC Plasma Source"
copyright = "2026, OpenMC Plasma Source contributors"
author = "OpenMC Plasma Source contributors"

extensions = ["sphinx_rtd_theme"]
root_doc = "index"
exclude_patterns = []
html_theme = "sphinx_rtd_theme"
html_title = "OpenMC Plasma Source Documentation"
html_short_title = "OpenMC Plasma Source"
html_logo = "_static/plasma_source_logo_v2.svg"
html_static_path = ["_static"]
html_css_files = ["theme_overrides.css"]
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "logo_only": False,
}
