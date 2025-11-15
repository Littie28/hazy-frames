Installation
============

Requirements
------------

- Python >= 3.12
- NumPy >= 2.3.4
- SciPy >= 1.16.3

Install from PyPI
-----------------

.. code-block:: bash

   pip install hazy-frames

Install with uv
---------------

.. code-block:: bash

   uv add hazy-frames

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
