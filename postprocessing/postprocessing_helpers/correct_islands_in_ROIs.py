########################################################################
###### IRINA GRIGORESCU
######
###### This is a helper script to detect and relabel disconnected island
###### components in a parcellation label GIFTI using majority voting
###### over bordering vertices.
######
########################################################################

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


# ---------------------------------------------------------------------------
# Adjacency builder
# ---------------------------------------------------------------------------

def build_adjacency_matrix(faces):
    """Build a symmetric sparse adjacency matrix from mesh faces."""
    nv = faces.max() + 1

    edge = np.concatenate([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], axis=0)

    A = coo_matrix(
        (np.ones(edge.shape[0], dtype=np.int8), (edge[:, 0], edge[:, 1])),
        shape=(nv, nv)
    )

    return (A + A.T).tocsr()


# ---------------------------------------------------------------------------
# Island detection & correction
# ---------------------------------------------------------------------------

def find_islands(labels, adjacency_matrix, min_island_size=2):
    """For each non-zero label, identify disconnected island components."""
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    islands = []

    for lab in unique_labels:
        verts = np.where(labels == lab)[0]
        if len(verts) == 0:
            continue

        subgraph = adjacency_matrix[verts][:, verts]
        n_components, component_ids = connected_components(subgraph, directed=False)

        if n_components == 1:
            continue

        groups = [verts[component_ids == c] for c in range(n_components)]
        groups.sort(key=len, reverse=True)
        main_verts = groups[0]
        island_groups = [g for g in groups[1:] if len(g) >= min_island_size]

        if not island_groups:
            continue

        islands.append({
            'label': int(lab),
            'main_verts': main_verts,
            'island_groups': island_groups,
        })

    return islands


def majority_neighbour_label(island_verts, labels, adjacency_matrix):
    """Find the most common non-background label among vertices bordering an island."""
    neighbour_rows = adjacency_matrix[island_verts]
    _, col_indices = neighbour_rows.nonzero()
    border_verts = np.unique(col_indices)
    border_verts = border_verts[~np.isin(border_verts, island_verts)]

    if len(border_verts) == 0:
        return 0

    border_labels = labels[border_verts]
    border_labels = border_labels[border_labels != 0]

    if len(border_labels) == 0:
        return 0

    values, counts = np.unique(border_labels, return_counts=True)
    return int(values[np.argmax(counts)])


def correct_islands(labels, adjacency_matrix, islands):
    """Relabel every island component to its majority surrounding label."""
    corrected = labels.copy()
    correction_log = []

    for entry in islands:
        for island_verts in entry['island_groups']:
            new_label = majority_neighbour_label(island_verts, corrected, adjacency_matrix)

            if new_label == 0:
                print(f"    [WARN] Label {entry['label']}: island of {len(island_verts)} "
                      f"vertices has no non-background neighbours — skipping.")
                continue

            corrected[island_verts] = new_label
            correction_log.append({
                'original_label': entry['label'],
                'new_label': new_label,
                'n_vertices': len(island_verts),
                'vertex_indices': island_verts.tolist(),
            })

    return corrected, correction_log


def print_summary(islands, correction_log, min_island_size):
    """Print summary statistics for island corrections."""
    print(f"\n{'='*60}")
    print(f"Island Correction Summary")
    print(f"Min island size corrected : {min_island_size}")
    print(f"{'='*60}")

    if not islands:
        print("  No islands found — all labels are fully connected. No changes made.")
        return

    total_island_verts = sum(
        len(g) for entry in islands for g in entry['island_groups']
    )
    print(f"  Labels with islands      : {len(islands)}")
    print(f"  Island components found  : {sum(len(e['island_groups']) for e in islands)}")
    print(f"  Total island vertices    : {total_island_verts}\n")

    print(f"  Corrections applied ({len(correction_log)}):")
    for c in correction_log:
        print(f"    Label {c['original_label']:>4} -> {c['new_label']:>4}  "
              f"({c['n_vertices']} vertices)")

    skipped = sum(len(e['island_groups']) for e in islands) - len(correction_log)
    if skipped:
        print(f"\n  {skipped} island(s) skipped (no non-background neighbours).")
    print()


# ---------------------------------------------------------------------------
# Primary Processing Function
# ---------------------------------------------------------------------------

def correct_islands_in_ROIs(subject_id, surface_hemi, input_dir, output_dir=None,
                            min_island_size=2, dry_run=False,
                            input_suffix="rois_filled",
                            output_suffix="rois_cleaned"):
    """
    Core function to load surface ROI files, detect disconnected label islands,
    relabel them using majority voting, and save the corrected label GIFTI file.
    """
    if output_dir is None:
        output_dir = input_dir

    # ------ construct paths ------
    labels_path  = os.path.join(input_dir, f"{subject_id}_pred_{surface_hemi}_{input_suffix}.label.gii")
    surface_path = os.path.join(input_dir, f"{subject_id}_pred_{surface_hemi}_white.surf.gii")
    output_path  = os.path.join(output_dir, f"{subject_id}_pred_{surface_hemi}_{output_suffix}.label.gii")

    # ------ print information ------
    print(f"\nRunning correct_islands_in_ROIs.py with:\n")
    print(f"SUBJECT     : {subject_id}")
    print(f"HEMISPHERE  : {surface_hemi}")
    print(f"INPUT DIR   : {input_dir}")
    print(f"OUTPUT DIR  : {output_dir}")
    print(f"MIN SIZE    : {min_island_size}")
    print(f"DRY RUN     : {dry_run}\n")

    # ------ validate inputs ------
    for path, label in [(labels_path, "Label file"), (surface_path, "Surface file")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    # ------ load data ------
    print("Loading files...")
    label_img   = nib.load(labels_path)
    labels      = label_img.darrays[0].data.astype(np.int32)
    surface_img = nib.load(surface_path)
    faces       = surface_img.darrays[1].data

    print(f"  Vertices      : {len(labels)}")
    print(f"  Unique labels : {len(np.unique(labels[labels != 0]))}")
    print(f"  Triangles     : {len(faces)}")

    # ------ build adjacency ------
    print("Building adjacency matrix...")
    adjacency_matrix = build_adjacency_matrix(faces)

    # ------ find islands ------
    print(f"Detecting islands (min island size = {min_island_size})...")
    islands = find_islands(labels, adjacency_matrix, min_island_size=min_island_size)

    if dry_run:
        print_summary(islands, [], min_island_size)
        print("  [Dry run] No output written.")
        return

    # ------ correct islands ------
    print("Correcting islands...")
    corrected_labels, correction_log = correct_islands(labels, adjacency_matrix, islands)
    print_summary(islands, correction_log, min_island_size)

    # ------ save output ------
    os.makedirs(output_dir, exist_ok=True)

    new_darray = nib.gifti.GiftiDataArray(
        data=corrected_labels.astype(np.int32),
        intent=label_img.darrays[0].intent,
        datatype=label_img.darrays[0].datatype,
        meta=label_img.darrays[0].meta,
    )

    new_img = nib.gifti.GiftiImage(
        header=label_img.header,
        extra=label_img.extra,
        meta=label_img.meta,
        labeltable=label_img.labeltable,
        darrays=[new_darray],
    )

    nib.save(new_img, output_path)
    print(f"  Corrected label file saved: {output_path}")


# ---------------------------------------------------------------------------
# Main CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Correct Disconnected Label Islands in ROIs")

    parser.add_argument('--subject_id', default=None, type=str,
                        help="Subject identifier, e.g. sub-001")
    parser.add_argument('--surface_hemi', default=None, type=str,
                        help="[left right]")
    parser.add_argument('--input_dir', default=None, type=str,
                        help="Path to input directory containing .gii files")
    parser.add_argument('--output_dir', default=None, type=str,
                        help="Path to output directory (default: same as input_dir)")
    parser.add_argument('--min_island_size', default=2, type=int,
                        help="Minimum vertices for island correction (default: 2)")
    parser.add_argument('--dry_run', action="store_true",
                        help="Detect and report islands without writing output")
    parser.add_argument('--input_suffix', default="rois_filled", type=str,
                        help="ROIs input suffix")
    parser.add_argument('--output_suffix', default="rois_cleaned", type=str,
                        help="ROIs input suffix")

    args = parser.parse_args()

    correct_islands_in_ROIs(
        subject_id=args.subject_id,
        surface_hemi=args.surface_hemi,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_island_size=args.min_island_size,
        dry_run=args.dry_run,
        input_suffix=args.input_suffix,
        output_suffix=args.output_suffix
    )

    print("Done\n")


if __name__ == "__main__":
    main()