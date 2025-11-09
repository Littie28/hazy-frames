from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

VSMALL: float = 1e-8
VVSMALL: float = 1e-12

IDENTITY_SCALE = np.ones(3, dtype=float)
IDENTITY_TRANSLATION = np.zeros(3, dtype=float)
IDENTITY_ROTATION = Rotation.identity()
