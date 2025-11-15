from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("../src"))

project = "hazy-frames"
copyright = "2025, Littie28"
author = "Littie28"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

nbsphinx_execute = "always"

nbsphinx_prolog = """
.. raw:: html

    <style>
        .nbinput .prompt,
        .nboutput .prompt {
            display: none;
        }
    </style>
"""

pygments_style = "sphinx"


def copy_notebooks(app, config):
    """Copy example notebooks from examples/ to docs/examples/ before build."""
    docs_dir = Path(app.confdir)
    source_dir = docs_dir.parent / "examples"
    target_dir = docs_dir / "examples"

    target_dir.mkdir(exist_ok=True)

    for notebook in source_dir.glob("*.ipynb"):
        target = target_dir / notebook.name
        shutil.copy2(notebook, target)
        print(f"Copied {notebook.name} to {target}")


def setup(app):
    app.connect("config-inited", copy_notebooks)
