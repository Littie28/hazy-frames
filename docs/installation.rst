Installation
============

Requirements
------------

- Python >= 3.12
- NumPy >= 2.3.4
- SciPy >= 1.16.3

Install from PyPI (not yet supported)
-------------------------------------

.. code-block:: bash

   pip install hazy-frames

Install with pip
----------------

.. code-block:: bash

   pip install git+https://github.com/Littie28/hazy-frames.git

Install with uv
---------------

.. code-block:: bash

   uv add git+https://github.com/Littie28/hazy-frames.git

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/Littie28/hazy-frames.git
   cd hazy-frames
   uv sync
   uv pip install -e .

Development installation
------------------------

To install with development dependencies:

.. code-block:: bash

   git clone https://github.com/Littie28/hazy-frames.git
   cd hazy-frames
   uv sync --group dev
