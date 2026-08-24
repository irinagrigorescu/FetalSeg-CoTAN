########################################################################
###### IRINA GRIGORESCU
######
###### This is a helper script to fill holes in a parcellation label GIFTI
###### using a binary ROI mask as ground truth for valid cortex.
######
########################################################################

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from scipy.sparse import coo_matrix
from collections import Counter
from nibabel import gifti


# ---------------------------------------------------------------------------
# Adjacency builder
# ---------------------------------------------------------------------------

def build_adjacency(faces):
    """
    Build a 1-hop vertex adjacency dict from mesh faces.

    Inputs:
        faces : (n_faces, 3) numpy array of vertex indices
    Returns:
        adjacency : dict mapping each vertex index -> set of neighbour indices
    """
    nv = faces.max() + 1

    # Stack all 3 directed edges from each triangle
    edge = np.concatenate([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], axis=0)

    # Build sparse adjacency matrix
    A = coo_matrix(
        (np.ones(edge.shape[0], dtype=np.int8), (edge[:, 0], edge[:, 1])),
        shape=(nv, nv)
    )

    # Convert sparse matrix to dict of sets for easy neighbour lookup
    adjacency = {i: set() for i in range(nv)}
    for src, tgt in zip(A.row, A.col):
        adjacency[src].add(tgt)
        adjacency[tgt].add(src)

    return adjacency


# ---------------------------------------------------------------------------
# Hole filling
# ---------------------------------------------------------------------------

def fill_label_holes(labels, mask, adjacency, max_iters=50):
    """
    Iteratively fill holes in a categorical label array using majority voting.
    """
    labels = labels.copy()

    for iteration in range(max_iters):
        holes = np.where(mask & (labels == 0))[0]

        if len(holes) == 0:
            print(f"  All holes filled after {iteration} iteration(s).")
            break

        print(f"  Iteration {iteration + 1}: {len(holes)} hole(s) remaining...")

        changed = False
        for v in holes:
            neighbour_labels = [
                labels[n] for n in adjacency[v] if labels[n] != 0
            ]

            if neighbour_labels:
                majority_label = Counter(neighbour_labels).most_common(1)[0][0]
                labels[v] = majority_label
                changed = True

        if not changed:
            remaining = np.sum(mask & (labels == 0))
            print(f"  Warning: no progress made. {remaining} hole(s) could "
                  f"not be filled (no labelled neighbours reachable).")
            break

    else:
        remaining = np.sum(mask & (labels == 0))
        if remaining > 0:
            print(f"  Warning: reached max_iters={max_iters}. "
                  f"{remaining} hole(s) still unfilled.")

    return labels


# ---------------------------------------------------------------------------
# GIFTI I/O helpers
# ---------------------------------------------------------------------------

def load_gifti_data(path, dtype=None):
    """Load a GIFTI file and return (gifti_img, data_array)."""
    img = nib.load(path)
    data = img.darrays[0].data
    if dtype is not None:
        data = data.astype(dtype)
    return img, data


def save_label_gifti(original_img, new_labels, output_path):
    """
    Save a label GIFTI (.label.gii), preserving the label table from
    the original image (parcel names, colours).
    """
    original_darray = original_img.darrays[0]

    new_darray = gifti.GiftiDataArray(
        data=new_labels.astype(np.int32),
        intent=1002,  # NIFTI_INTENT_LABEL
        datatype=16,  # NIFTI_TYPE_INT32
        meta=original_darray.meta,
    )

    new_img = gifti.GiftiImage(
        header=original_img.header,
        meta=original_img.meta,
        darrays=[new_darray],
        labeltable=original_img.labeltable,
    )

    nib.save(new_img, output_path)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Primary Processing Function
# ---------------------------------------------------------------------------

def fill_holes_in_rois(subject_id, surface_hemi, input_dir, output_dir,
                       input_suffix="rois", output_suffix="rois_filled", max_iters=50):
    """
    Core function to load surface ROI files, fill unlabelled hole vertices,
    and save the cleaned label GIFTI output.
    """
    # ------ construct paths ------
    mask_path    = os.path.join(input_dir, f"{subject_id}_pred_{surface_hemi}_roi.shape.gii")
    labels_path  = os.path.join(input_dir, f"{subject_id}_pred_{surface_hemi}_{input_suffix}.label.gii")
    surface_path = os.path.join(input_dir, f"{subject_id}_pred_{surface_hemi}_white.surf.gii")
    output_path  = os.path.join(output_dir, f"{subject_id}_pred_{surface_hemi}_{output_suffix}.label.gii")

    # ------ print information ------
    print(f"\nRunning fill_holes_in_ROIs.py with:\n")
    print(f"SUBJECT     : {subject_id}")
    print(f"HEMISPHERE  : {surface_hemi}")
    print(f"INPUT DIR   : {input_dir}")
    print(f"OUTPUT DIR  : {output_dir}\n")

    # ------ validate inputs ------
    for path, label in [(mask_path, "Binary mask"), (labels_path, "Label file"), (surface_path, "Surface file")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)
        print(f"  Found {label}: {path}")

    os.makedirs(output_dir, exist_ok=True)

    # ------ load data ------
    print("\nLoading files...")
    label_img, labels = load_gifti_data(labels_path, dtype=np.int32)
    _, mask = load_gifti_data(mask_path, dtype=np.float32)
    surface_img = nib.load(surface_path)

    mask = mask.astype(bool)

    print(f"  Vertices  : {len(labels)}")
    print(f"  Mask      : {mask.sum()} valid cortex vertices")
    print(f"  Holes     : {np.sum(mask & (labels == 0))} vertices to fill")

    # ------ build adjacency ------
    print("\nBuilding vertex adjacency...")
    faces = surface_img.darrays[1].data
    adjacency = build_adjacency(faces)
    print(f"  Triangles : {len(faces)}")

    # ------ fill holes ------
    print("\nFilling holes...")
    filled_labels = fill_label_holes(labels, mask, adjacency, max_iters=max_iters)

    # ------ zero out non-cortex vertices ------
    print("\nMasking out vertices outside binary ROI...")
    before = np.sum(filled_labels != 0)
    filled_labels[~mask] = 0
    after = np.sum(filled_labels != 0)
    print(f"  Vertices zeroed out: {before - after}")

    # ------ sanity check ------
    remaining_holes = np.sum(mask & (filled_labels == 0))
    print(f"\nHoles remaining after fill: {remaining_holes}")
    if remaining_holes > 0:
        print("  Warning: some holes could not be filled (isolated vertices).")

    # ------ save output ------
    print("\nSaving output...")
    save_label_gifti(label_img, filled_labels, output_path)


# ---------------------------------------------------------------------------
# Main CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fill Holes in Parcellation ROIs")

    parser.add_argument('--subject_id', default=None, type=str,
                        help="Subject identifier, e.g. sub-001")
    parser.add_argument('--surface_hemi', default=None, type=str,
                        help="[left right]")
    parser.add_argument('--input_dir', default=None, type=str,
                        help="Path to input directory containing .gii files")
    parser.add_argument('--output_dir', default=None, type=str,
                        help="Path to output directory to save cleaned label file")
    parser.add_argument('--max_iters', default=50, type=int,
                        help="Maximum fill iterations (default: 50)")
    parser.add_argument('--input_suffix', default="rois", type=str,
                        help="ROIs input suffix")
    parser.add_argument('--output_suffix', default="rois_filled", type=str,
                        help="ROIs input suffix")

    args = parser.parse_args()

    fill_holes_in_rois(
        subject_id=args.subject_id,
        surface_hemi=args.surface_hemi,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_iters=args.max_iters,
        input_suffix=args.input_suffix,
        output_suffix=args.output_suffix
    )

    print("Done\n")


if __name__ == "__main__":
    main()
