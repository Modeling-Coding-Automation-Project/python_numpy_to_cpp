from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np


time = np.array([[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
series_data = np.sin(2 * np.pi * time)

print("time =")
print(time)
print("\n")
print("series_data =")
print(series_data)
