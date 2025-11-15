Quick Start
===========

Basic Usage
-----------

Creating Frames
^^^^^^^^^^^^^^^

Every coordinate transformation starts with a root frame:

.. code-block:: python

   from hazy import Frame

   world = Frame.make_root("world")

Create child frames with transformations:

.. code-block:: python

   camera = (
       world.make_child("camera")
       .translate(x=1, y=2, z=5)
       .rotate_euler(y=90, degrees=True)
   )

Working with Points
^^^^^^^^^^^^^^^^^^^

Create points in a frame:

.. code-block:: python

   point = world.point(1, 2, 3)

Transform points between frames:

.. code-block:: python

   point_in_camera = point.to_frame(camera)

Working with Vectors
^^^^^^^^^^^^^^^^^^^^

Vectors are direction quantities (unaffected by translation):

.. code-block:: python

   direction = world.vector(0, 0, 1)  # Looking down Z-axis
   direction_in_camera = direction.to_frame(camera)

Type Safety
-----------

Points and vectors are distinct types with mathematically valid operations:

.. code-block:: python

   # Valid operations
   v1 + v2          # Vector addition
   point + vector   # Point displacement
   point - point    # Vector between points
   vector * 2       # Scalar multiplication

   # Invalid operations (raise TypeError)
   point + point    # Can't add points
   point * 2        # Can't scale points

Convert to NumPy
----------------

Opt out to raw arrays when needed:

.. code-block:: python

   import numpy as np

   coords = np.array(point)  # [x, y, z]
   point * 2  # TypeError
   np.array(point) * 2  # Works: [2x, 2y, 2z]

Next Steps
----------

- See :doc:`examples/index` for detailed examples
- Read the :doc:`api/index` for complete API reference
