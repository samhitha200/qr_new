Python 3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy

def extract_white_area_features(image):
    features = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to isolate white areas
    _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    white_pixels = gray[white_mask == 255]

    if white_pixels.size == 0:
        features.extend([0] * 18)
...         return np.array(features)
... 
...     white_ratio = white_pixels.size / gray.size
... 
...     # Intensity features
...     features.append(np.mean(white_pixels))
...     features.append(np.std(white_pixels))
...     features.append(np.min(white_pixels))
...     features.append(np.max(white_pixels))
...     features.append(white_ratio)
... 
...     # LBP texture features
...     lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
...     lbp_masked = lbp[white_mask == 255]
...     lbp_hist, _ = np.histogram(lbp_masked, bins=np.arange(0, 11), range=(0, 10))
...     lbp_hist = lbp_hist.astype("float")
...     lbp_hist /= (lbp_hist.sum() + 1e-6)
...     features.extend(lbp_hist.tolist())
... 
...     # Entropy
...     entropy_val = shannon_entropy(white_pixels)
...     features.append(entropy_val)
... 
...     # Laplacian variance (sharpness)
...     laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
...     features.append(laplacian_var)
... 
...     # FFT-based high frequency energy (screen pattern)
...     f = np.fft.fft2(gray)
...     fshift = np.fft.fftshift(f)
...     magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
...     center_crop = magnitude_spectrum[
...         gray.shape[0]//4 : 3*gray.shape[0]//4,
...         gray.shape[1]//4 : 3*gray.shape[1]//4
...     ]
...     high_freq_energy = np.mean(center_crop)
...     features.append(high_freq_energy)
... 
...     return np.array(features)
