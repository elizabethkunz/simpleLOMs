# Configuration file for the Sphinx documentation builder.
# See https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src"))

project = "simpleLOMs"
copyright = "2025, Elizabeth Kunz"
author = "Elizabeth Kunz"
release = "0.1.0"
version = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".venv", "**/.venv"]

# --- HTML output / theme ---------------------------------------------------
# Furo: a clean, minimalist theme with a native light/dark toggle and a
# right-hand "On this page" table of contents. Falls back gracefully if the
# theme is not installed in the build environment.
html_theme = "furo"
try:
    import furo  # noqa: F401
except ImportError:
    html_theme = "alabaster"

html_static_path = ["_static"]
html_css_files = ["custom.css"]
# Furo picks the logo per color mode via ``light_logo`` / ``dark_logo`` in
# ``html_theme_options`` below (paths are relative to ``_static``), so no
# single ``html_logo`` is set here.
html_favicon = "_static/logo-light.png"
html_title = "simpleLOMs"
html_short_title = "simpleLOMs"

# Cool, cryogenic physics palette: superconducting cyan + a quantum-violet
# secondary accent, tuned separately for light and dark backgrounds.
html_theme_options = {
    "sidebar_hide_name": False,
    "light_logo": "logo-light.png",
    "dark_logo": "logo-dark.png",
    "light_css_variables": {
        "color-brand-primary": "#0e7490",
        "color-brand-content": "#0e7490",
        "color-brand-visited": "#7c3aed",
        "color-background-secondary": "#f6f9fb",
        "color-api-name": "#0e7490",
        "color-api-pre-name": "#7c3aed",
        "color-highlight-on-target": "#e0f2fe",
        "font-stack": ("Inter, -apple-system, BlinkMacSystemFont, "
                       "'Segoe UI', Roboto, Helvetica, Arial, sans-serif"),
        "font-stack--monospace": ("'JetBrains Mono', 'SF Mono', 'Fira Code', "
                                  "Menlo, Consolas, monospace"),
    },
    "dark_css_variables": {
        "color-brand-primary": "#22d3ee",
        "color-brand-content": "#38bdf8",
        "color-brand-visited": "#a78bfa",
        "color-background-primary": "#0b0f14",
        "color-background-secondary": "#0f151c",
        "color-background-hover": "#151d27",
        "color-api-name": "#22d3ee",
        "color-api-pre-name": "#a78bfa",
        "color-highlight-on-target": "#13303a",
    },
    "footer_icons": [
        {
            "name": "Levenson-Falk Lab",
            "url": "https://dornsife.usc.edu/lfl/",
            "html": (
                '<span style="font-weight:600;font-size:0.85rem;">LFL</span>'
            ),
            "class": "",
        },
        {
            "name": "LFL Lab GitHub",
            "url": "https://github.com/LFL-Lab",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" '
                'stroke-width="0" viewBox="0 0 16 16" aria-hidden="true">'
                '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 '
                '2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49'
                '-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15'
                '-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33'
                '.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31'
                '-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64'
                '-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2'
                '-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87'
                ' 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 '
                '.21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z">'
                '</path></svg>'
            ),
            "class": "",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/elizabethkunz/simpleLOMs",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" '
                'stroke-width="0" viewBox="0 0 16 16">'
                '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 '
                '2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49'
                '-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15'
                '-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33'
                '.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31'
                '-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64'
                '-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2'
                '-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87'
                ' 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 '
                '.21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z">'
                '</path></svg>'
            ),
            "class": "",
        },
    ],
}

# Render notebook source without executing cells (avoids import/runtime failures
# when simpleLOMs or optional deps are not installed in the doc-build env).
nbsphinx_execute = "never"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}
