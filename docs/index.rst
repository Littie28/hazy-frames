hazy-frames
===========

Crystal clear coordinate transformations for Python.

**hazy-frames** is a lightweight library for managing hierarchical coordinate frames and transforming points and vectors between them. Built for the NumPy and SciPy ecosystems with a pythonic, chainable API.

.. code-block:: python

   from hazy import Frame, Point, Vector

   # Create coordinate frames with chaining
   world = Frame.make_root(name="world")
   camera = (
       world.make_child("camera")
       .translate(z=5)
       .rotate_euler(y=90, degrees=True)
       .freeze()
   )

   # Transform points between frames
   point_world = world.point(1, 2, 3)
   point_camera = point_world.to_frame(camera)

Features
--------

- **Type Safety**: Points and Vectors are distinct types
- **Mathematical Correctness**: Only formally valid operations allowed
- **Chainable API**: Fluent method chaining for frame transformations
- **Zero Runtime Overhead**: Uses views where possible, avoids unnecessary copies
- **Easy Opt-Out**: Convert to numpy arrays when you need raw manipulation

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   examples/index
   api/index

Installation
------------

.. code-block:: bash

   pip install git+https://github.com/Littie28/hazy-frames.git

Or with uv:

.. code-block:: bash

   uv add git+https://github.com/Littie28/hazy-frames.git

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
