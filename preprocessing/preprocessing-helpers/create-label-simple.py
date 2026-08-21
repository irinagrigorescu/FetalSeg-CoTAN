import nibabel as nib
import numpy as np
import sys
from scipy.ndimage import gaussian_filter

# ---- Inputs ----
lab43_path  = sys.argv[1]
output_path = sys.argv[2]
mask_path   = sys.argv[3]

# ---- Load volumes ----
img_lab43 = nib.load(lab43_path)
lab43 = img_lab43.get_fdata()

# Ensure integer type
lab43 = lab43.astype(np.int32)

# ---- Create output ----
out = np.zeros_like(lab43, dtype=np.int32)

# ---- Define label groups ----
csf_label = 1
cgm_label = 2
wm_label  = 3
cerebellum_label = 4
brainstem_label = 5

# LAB43 CSF labels
csf_labels = np.array([37, 38, 43])

# LAB43 CGM hemisphere labels
cgm_labels = np.array([1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12])

# LAB43 WM hemisphere labels
wm_labels = np.array([14,16,18,20,22,24, 13,15,17,19,21,23, 40, 39, 41, 31,33,35, 32,34,36, 42, 26, 25])

# LAB43 brainstem and cerebellum
brainstem_labels = np.array([27])
cerebellum_labels = np.array([28, 29, 30])

# ---- Assign labels ----
out[np.isin(lab43, csf_labels)] = csf_label
out[np.isin(lab43, cgm_labels)] = cgm_label
out[np.isin(lab43, wm_labels)] = wm_label
out[np.isin(lab43, cerebellum_labels)] = cerebellum_label
out[np.isin(lab43, brainstem_labels)] = brainstem_label

# ---- Create binary mask of the brain ----
binary_mask = (lab43 > 0).astype(np.float32)
smoothed_mask = gaussian_filter(binary_mask, sigma=2.0)
binary_mask_smoothed = (smoothed_mask >= 0.5).astype(np.int32)

# ---- Save Label Map ----
out_img = nib.Nifti1Image(out, affine=img_lab43.affine, header=img_lab43.header)
nib.save(out_img, output_path)
print(f"[+] PYTHON  saved simple labels to: {output_path}")

# ---- Save Label Map ----
mask_img = nib.Nifti1Image(binary_mask_smoothed, affine=img_lab43.affine, header=img_lab43.header)
nib.save(mask_img, mask_path)
print(f"[+] PYTHON  saved brain mask    to: {mask_path}")

