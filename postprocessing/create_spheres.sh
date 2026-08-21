#!/bin/bash

set -euo pipefail

# -------------- Argument Validation
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <SUBJ> <SUBJ_DIR> <HEMISPHERE: left|right>" >&2
    exit 1
fi

# Arguments
SUBJ="$1"
SUBJ_DIR="$2"
HEMI="$3"

# Suffixes needed for freesurfer
FS_HEMI=$([[ "$HEMI" == "left" ]] && echo "lh" || echo "rh")

# Input / output files
INSURF_IN="${SUBJ_DIR}/${SUBJ}_pred_${HEMI}_white.surf.gii"
SPHERE_OUT="${SUBJ_DIR}/${SUBJ}_pred_${HEMI}_sphere.surf.gii"
INFLATED_OUT="${SUBJ_DIR}/${SUBJ}_pred_${HEMI}_inflated.surf.gii"
VINFLATED_OUT="${SUBJ_DIR}/${SUBJ}_pred_${HEMI}_vinflated.surf.gii"

# Skip if target surface files already exist
if [[ -f "$SPHERE_OUT" ]] && [[ -f "$INFLATED_OUT" ]] && [[ -f "$VINFLATED_OUT" ]]; then
    echo "[+] Inflation outputs already exist for ${SUBJ} (${HEMI}). Skipping."
    exit 0
fi

if [[ ! -f "$INSURF_IN" ]]; then
    echo "[-] Input surface missing: ${INSURF_IN}" >&2
    exit 1
fi

# The temporary files stored by freesurfer while creating spheres/inflated/v inflated
TMP_INSURF="${SUBJ_DIR}/${FS_HEMI}.white"
TMP_SMOOTH="${SUBJ_DIR}/${FS_HEMI}.smoothwm"
TMP_INFLATED="${SUBJ_DIR}/${FS_HEMI}.inflated"
TMP_VINFLATED="${SUBJ_DIR}/${FS_HEMI}.vinflated"
TMP_INFLATED_SPHERE="${SUBJ_DIR}/${FS_HEMI}.inflated4sphere"
TMP_SPHERE="${SUBJ_DIR}/${FS_HEMI}.sphere"
TMP_SULC="${SUBJ_DIR}/${FS_HEMI}.sulc"

# Load FreeSurfer
source ~/.bashrc

echo "[*] Processing FreeSurfer inflation/sphere generation for ${SUBJ} (${HEMI})..."

# -------------- FreeSurfer Pipeline
# converts my prediction into something for freesurfer - no suffix
mris_convert "$INSURF_IN" "$TMP_INSURF"
# smooths it
mris_smooth -n 3 -nw -seed 1234 "$TMP_INSURF" "$TMP_SMOOTH"
# inflates it for sphere
mris_inflate -seed 1234 ${TMP_SMOOTH} ${TMP_INFLATED_SPHERE} -no-save-sulc

# inflates it to create the inflated surface
mris_inflate -seed 1234 -n 1 ${TMP_SMOOTH} ${TMP_INFLATED} -no-save-sulc
mris_smooth -n 3 -nw -seed 1234 ${TMP_INFLATED} ${TMP_INFLATED}
for i in 0 1 2 ; do
    mris_inflate -seed 1234 -n 1 ${TMP_INFLATED} ${TMP_INFLATED} -no-save-sulc
    mris_smooth -n 3 -nw -seed 1234 ${TMP_INFLATED} ${TMP_INFLATED}
done

# inflates it further to create the very inflated surface
mris_inflate -seed 1234 -n 4 ${TMP_INFLATED} ${TMP_VINFLATED} -no-save-sulc
mris_smooth -n 3 -nw -seed 1234 ${TMP_VINFLATED} ${TMP_VINFLATED}

# make it into a sphere
mris_sphere -seed 1234 "$TMP_INFLATED_SPHERE" "$TMP_SPHERE"

# convert back to surf.gii
mris_convert "$TMP_SPHERE" "$SPHERE_OUT"
mris_convert "$TMP_INFLATED" "$INFLATED_OUT"
mris_convert "$TMP_VINFLATED" "$VINFLATED_OUT"

rm -f "$TMP_INSURF" "$TMP_SMOOTH" "$TMP_INFLATED" "$TMP_INFLATED_SPHERE" "$TMP_VINFLATED" "$TMP_SPHERE" "$TMP_SULC"

# -------------- Final Output Verification Check
missing_files=0

if [[ ! -f "$SPHERE_OUT" ]]; then
    echo "[-] Error: Expected output sphere surface missing: ${SPHERE_OUT}" >&2
    missing_files=1
fi

if [[ ! -f "$INFLATED_OUT" ]]; then
    echo "[-] Error: Expected output inflated surface missing: ${INFLATED_OUT}" >&2
    missing_files=1
fi

if [[ ! -f "$VINFLATED_OUT" ]]; then
    echo "[-] Error: Expected output vinflated surface missing: ${VINFLATED_OUT}" >&2
    missing_files=1
fi

if [[ "$missing_files" -ne 0 ]]; then
    echo "[-] Error: One or more surface outputs were not successfully created." >&2
    exit 1
fi

echo "[+] Sphere generation complete: ${SPHERE_OUT}"
