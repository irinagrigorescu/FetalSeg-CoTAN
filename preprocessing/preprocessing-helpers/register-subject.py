#!/usr/bin/env python

import os
import argparse
import ants
import numpy as np
from scipy.io import loadmat
import nibabel as nib


def affine_matrix(ants_trans):
    """
    Convert ants transform to a 4x4 affine matrix - check also https://github.com/m-qiang/CoTAN
    """
    transform = np.zeros([4, 4])
    m_matrix = loadmat(
        ants_trans['fwdtransforms'][0])['AffineTransform_float_3_3'][:9].reshape(3, 3)  # .T
    m_center = loadmat(
        ants_trans['fwdtransforms'][0])['fixed'][:, 0]
    m_translate = loadmat(
        ants_trans['fwdtransforms'][0])['AffineTransform_float_3_3'][9:][:, 0]
    m_offset = m_translate + m_center - m_matrix @ m_center

    print(m_matrix)

    # ITK affine to affine matrix
    transform[:3, :3] = m_matrix
    transform[:3, -1] = -m_offset
    transform[3, :] = np.array([0, 0, 0, 1])

    # LIP space to RAS
    transform[2, -1] = -transform[2, -1]
    transform[2, 1] = -transform[2, 1]
    transform[1, 2] = -transform[1, 2]
    transform[2, 0] = -transform[2, 0]
    transform[0, 2] = -transform[0, 2]

    return transform


def get_affine_out_path(input_path, output_directory):
    base_name = os.path.basename(input_path)

    # Handle standard NIfTI file extensions
    if base_name.endswith('.nii.gz'):
        new_name = base_name.replace('.nii.gz', '_affine.nii.gz')
    elif base_name.endswith('.nii'):
        new_name = base_name.replace('.nii', '_affine.nii')
    else:
        new_name = f"{base_name}_affine"

    return os.path.join(output_directory, new_name)


def main():
    parser = argparse.ArgumentParser(description="Affine registration using custom pipeline")
    parser.add_argument("--subj", required=True, help="Subject")
    parser.add_argument("--t2_path", required=True, help="Path to input T2w brain image")
    parser.add_argument("--label_path", required=True, help="Path to main label image")
    parser.add_argument("--label_simple_path", required=True, help="Path to simplified label image")
    parser.add_argument("--mask_path", required=True, help="Path to brain mask")
    parser.add_argument("--template_path", required=True, help="Path to reference template image")
    parser.add_argument("--out_dir", required=True, help="Output directory")

    args = parser.parse_args()

    subj = args.subj
    t2_path = args.t2_path
    label_path = args.label_path
    label_simple_path = args.label_simple_path
    mask_path = args.mask_path
    template_path = args.template_path
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    # Load images
    # -----------------------------
    print("Loading images...")
    mask_brain = ants.image_read(mask_path)
    moving = ants.image_read(t2_path)
    lab_main = ants.image_read(label_path)
    lab_simple = ants.image_read(label_simple_path)
    fixed = ants.image_read(template_path)

    # Pre-mask the brain
    moving = moving * mask_brain

    # template affine
    affine_fix = nib.load(template_path).affine

    # -----------------------------
    # Registration
    # -----------------------------
    print("Running registration...")
    # affine registration
    ants_trans = ants.registration(
        fixed=fixed,
        moving=moving,
        outprefix=os.path.join(out_dir, subj) + '_affine_',
        type_of_transform='Affine',
        aff_metric='mattes')

    # -----------------------------
    # Transform main image and main label
    # -----------------------------
    # warp the image
    img_aligned = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=ants_trans['fwdtransforms'],
        interpolator='linear')
    # warp the label
    lab_simple_aligned = ants.apply_transforms(
        fixed=fixed,
        moving=lab_simple,
        transformlist=ants_trans['fwdtransforms'],
        interpolator='genericLabel')
    # warp the label
    lab_main_aligned = ants.apply_transforms(
        fixed=fixed,
        moving=lab_main,
        transformlist=ants_trans['fwdtransforms'],
        interpolator='genericLabel')

    # compute new affine matrix
    affine_mat = affine_matrix(ants_trans)
    affine_warp = affine_mat @ affine_fix

    # -----------------------------
    # Save aligned T2 and labels
    # -----------------------------
    # 1. Aligned T2
    out_t2_path = get_affine_out_path(t2_path, out_dir)
    warp_img_nib = nib.Nifti1Image(img_aligned.numpy().astype(np.float32), affine_warp)
    warp_img_nib.header['xyzt_units'] = 2
    nib.save(warp_img_nib, out_t2_path)
    print(f"[+] Saved aligned T2 to: {out_t2_path}")

    # 2. Aligned Simple Label
    out_lab_simple_path = get_affine_out_path(label_simple_path, out_dir)
    warp_img_nib = nib.Nifti1Image(lab_simple_aligned.numpy().astype(np.float32), affine_warp)
    warp_img_nib.header['xyzt_units'] = 2
    nib.save(warp_img_nib, out_lab_simple_path)
    print(f"[+] Saved aligned simple label to: {out_lab_simple_path}")

    # 3. Aligned Main Label
    out_lab_main_path = get_affine_out_path(label_path, out_dir)
    warp_img_nib = nib.Nifti1Image(lab_main_aligned.numpy().astype(np.float32), affine_warp)
    warp_img_nib.header['xyzt_units'] = 2
    nib.save(warp_img_nib, out_lab_main_path)
    print(f"[+] Saved aligned main label to: {out_lab_main_path}")

    print("Done.")


if __name__ == "__main__":
    main()
