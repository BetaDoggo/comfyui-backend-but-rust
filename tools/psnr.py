"""PSNR between two images. usage: python psnr.py a.png b.png"""
import sys

import numpy as np
from PIL import Image

a, b = (np.asarray(Image.open(p).convert("RGB")).astype(np.float64) for p in sys.argv[1:3])
mse = ((a - b) ** 2).mean()
print(f"PSNR {10 * np.log10(255 ** 2 / mse) if mse else float('inf'):.2f} dB, mean abs {np.abs(a - b).mean():.3f}, max {np.abs(a - b).max():.0f}")
