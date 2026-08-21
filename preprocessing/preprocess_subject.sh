#!/bin/bash

set -euo pipefail

######
###### Irina Grigorescu | 2026-04-22
###### This script takes as input a subject and prepares it for inference by:
###### 1. Generating simplified labels and a brain mask.
###### 2. Affinely registering the subject data to a fetal dHCP template.
######
###### Usage:
######   ./preprocess-subject.sh <SUBJ> <PATH/TO/INPUT/FOLDER> <PATH/TO/OUTPUT/FOLDER>
######
###### Input Parameters:
######   ${SUBJ}                 : Subject ID (e.g., sub-01_ses-01)
######   PATH/TO/INPUT/FOLDER    : Directory containing original subject subfolders
######   PATH/TO/OUTPUT/FOLDER   : Directory where affine-registered outputs will be saved
######
###### Expected Input Data:
######   PATH/TO/INPUT/FOLDER/${SUBJ}/
######     ├── ${SUBJ}_T2w.nii.gz
######     └── ${SUBJ}_LAB43_brain.nii.gz           (This is the multi-bounti data)
######
###### Generated Outputs:
###### 1. Preprocessing outputs generated inside input directory:
######   PATH/TO/INPUT/FOLDER/${SUBJ}/
######     ├── ${SUBJ}_T2w.nii.gz
######     ├── ${SUBJ}_LAB43_brain.nii.gz
######     ├── ${SUBJ}_LAB_brain.nii.gz             (Simplified labels: 1-csf|2-cgm|3-wm|4-cerebellum|5-brainstem)
######     └── ${SUBJ}_brain-mask.nii.gz            (Smoothed binary brain mask)
######
###### 2. Affine-registered outputs saved in output directory:
######   PATH/TO/OUTPUT/FOLDER/${SUBJ}/
######     ├── ${SUBJ}_affine_0GenericAffine.mat   (Affine transformation matrix)
######     ├── ${SUBJ}_LAB_brain_affine.nii.gz      (Affine-aligned simplified label)
######     ├── ${SUBJ}_LAB43_brain_affine.nii.gz    (Affine-aligned LAB43 label)
######     └── ${SUBJ}_T2w_affine.nii.gz            (Affine-aligned T2w image)
######
###### Dependencies & Helpers:
######   - Template: ../templates/dhcp_fetal_week36_t2w.nii.gz
######   - Helper 1: preprocessing_helpers/create_label_simple.py
######   - Helper 2: preprocessing_helpers/register_subject.py
######
###### Execution Steps:
###### 1. Verify that required input files exist.
###### 2. Generate simplified label map and binary mask if not already present.
###### 3. Ensure subject output directory exists.
###### 4. Perform affine registration to fetal template if output files do not already exist.
######

# -------------- CHECK ARGUMENTS
if [ "$#" -ne 3 ]; then
    echo "Error: Expected 3 arguments, but got $#" >&2
    echo "Usage: $0 <SUBJ> <PATH/TO/INPUT/FOLDER> <PATH/TO/OUTPUT/FOLDER>" >&2
    exit 1
fi
# -------------- ASSIGN ARGUMENTS
SUBJ="$1"
INPUT_DIR="$2"
OUTPUT_DIR="$3"

# INPUT PATHS
FOLDER_ORIG="${INPUT_DIR}/${SUBJ}/"   # e.g.: ${SUBJ}_T2w.nii.gz | ${SUBJ}_LAB43_brain.nii.gz

FILE_T2W_ORIG="${FOLDER_ORIG}/${SUBJ}_T2w.nii.gz"
FILE_SEG_ORIG="${FOLDER_ORIG}/${SUBJ}_LAB43_brain.nii.gz"
FILE_SEG_SIMPLE="${FOLDER_ORIG}/${SUBJ}_LAB_brain.nii.gz"
FILE_MASK_SIMPLE="${FOLDER_ORIG}/${SUBJ}_brain-mask.nii.gz"

# OUTPUT PATHS
FOLDER_OUTPUT_SUBJECT="${OUTPUT_DIR}/${SUBJ}"
TEMPLATE_PATH="../templates/dhcp_fetal_week36_t2w.nii.gz"

# Make sure subject output directory exists
mkdir -p "${FOLDER_OUTPUT_SUBJECT}"

# -------------- Check both files exist
if [[ -f "${FILE_T2W_ORIG}" ]] && [[ -f "${FILE_SEG_ORIG}" ]]; then
    echo -e "[+] SUBJECT ${SUBJ} files exist."
else
    [[ ! -f "${FILE_T2W_ORIG}" ]] && echo -e "[-] Missing file: ${FILE_T2W_ORIG}" >&2
    [[ ! -f "${FILE_SEG_ORIG}" ]] && echo -e "[-] Missing file: ${FILE_SEG_ORIG}" >&2
    exit 2
fi

# -------------- Transform LAB43 into LAB (simple) and create brain mask
if [[ -f "${FILE_SEG_SIMPLE}" ]] && [[ -f "${FILE_MASK_SIMPLE}" ]]; then
    echo -e "[+] Simplified label and brain mask already exist for ${SUBJ}. Skipping generation."
else
    echo -e "[*] Generating simplified label and brain mask..."
    python preprocessing_helpers/create_label_simple.py \
           "${FILE_SEG_ORIG}" \
           "${FILE_SEG_SIMPLE}" \
           "${FILE_MASK_SIMPLE}"
fi

# -------------- Perform affine transformation from T2w brain to T2w fetal dHCP template
FILE_AFFINE_MAT="${FOLDER_OUTPUT_SUBJECT}/${SUBJ}_affine_0GenericAffine.mat"
FILE_AFFINE_LAB="${FOLDER_OUTPUT_SUBJECT}/${SUBJ}_LAB_brain_affine.nii.gz"
FILE_AFFINE_LAB43="${FOLDER_OUTPUT_SUBJECT}/${SUBJ}_LAB43_brain_affine.nii.gz"
FILE_AFFINE_T2W="${FOLDER_OUTPUT_SUBJECT}/${SUBJ}_T2w_affine.nii.gz"

# -------------- Check if outputs exist first
if [[ -f "${FILE_AFFINE_MAT}" ]] && \
   [[ -f "${FILE_AFFINE_LAB}" ]] && \
   [[ -f "${FILE_AFFINE_LAB43}" ]] && \
   [[ -f "${FILE_AFFINE_T2W}" ]]; then
    echo -e "[+] All affine registration outputs already exist for ${SUBJ}. Skipping registration."
else
    echo -e "[*] Running affine registration for ${SUBJ}..."
    python preprocessing_helpers/register_subject.py \
            --subj "${SUBJ}" \
            --t2_path "${FILE_T2W_ORIG}" \
            --label_path "${FILE_SEG_ORIG}" \
            --label_simple_path "${FILE_SEG_SIMPLE}" \
            --mask_path "${FILE_MASK_SIMPLE}" \
            --template_path "${TEMPLATE_PATH}" \
            --out_dir "${FOLDER_OUTPUT_SUBJECT}"
fi
