#!/bin/bash

######
###### Irina Grigorescu | 2026-04-22
###### This script reads a TSV file of subjects and runs the subject preparation
###### pipeline for each subject using preprocess-subject.sh. It verifies that
###### the expected affine-aligned outputs were successfully created and records
###### successful and failed subjects in separate TSV files within the output directory.
######
###### Usage:
######   ./run_pipeline.sh <PATH/TO/TSV> <PATH/TO/INPUT/FOLDER> <PATH/TO/OUTPUT/FOLDER>
######
###### Input Parameters:
######   1. TSV_FILE      : Path to the input TSV file containing subject metadata
######   2. FOLDER_INPUT  : Path to the root input directory containing original subject data
######   3. FOLDER_OUTPUT : Path to the root output directory where results and logs are saved
######
###### Expected TSV Format:
######   participant_id   session_id   scan_age
######
###### Per-Subject Pipeline Invocation:
######   SUBJ="sub-${participant_id}_ses-${session_id}"
######   ./preprocess-subject.sh ${SUBJ} ${FOLDER_INPUT} ${FOLDER_OUTPUT}
######
###### Output Directory Structure (${FOLDER_OUTPUT}/${SUBJ}/):
######     ├── ${SUBJ}_affine_0GenericAffine.mat
######     ├── ${SUBJ}_T2w_affine.nii.gz
######     ├── ${SUBJ}_LAB_brain_affine.nii.gz
######     └── ${SUBJ}_LAB43_brain_affine.nii.gz
######
###### Logs Generated in ${FOLDER_OUTPUT}:
######     ├── preprocessing-success.tsv
######     └── preprocessing-failed.tsv
######
###### Steps:
###### 1. Validate that exactly 3 command-line arguments are provided
###### 2. Verify that the input TSV file exists
###### 3. Ensure the output folder exists and initialize success/failed log headers
###### 4. Read subjects line-by-line from the TSV file
###### 5. Construct subject ID as sub-${participant_id}_ses-${session_id}
###### 6. Run preprocess-subject.sh ${SUBJ} ${FOLDER_INPUT} ${FOLDER_OUTPUT}
###### 7. Check if all required output files exist in ${FOLDER_OUTPUT}/${SUBJ}/
###### 8. Log subject result to preprocessing-success.tsv or preprocessing-failed.tsv
###### 9. Output final count summary of total, successful, and failed subjects
######

# -----------------------------
# CHECK & ASSIGN ARGUMENTS
# -----------------------------
if [ "$#" -ne 3 ]; then
    echo "Error: Expected 3 arguments, but got $#" >&2
    echo "Usage: $0 <PATH/TO/TSV> <PATH/TO/INPUT/FOLDER> <PATH/TO/OUTPUT/FOLDER>" >&2
    exit 1
fi

TSV="$1"
FOLDER_INPUT="$2"
FOLDER_OUTPUT="$3"

# Validate TSV file existence
if [[ ! -f "$TSV" ]]; then
    echo "Error: Input TSV file does not exist: $TSV" >&2
    exit 1
fi

SCRIPT="preprocess-subject.sh"

# Ensure output directory exists for logs
mkdir -p "$FOLDER_OUTPUT"

LOG_SUCCESS="${FOLDER_OUTPUT}/preprocessing-success.tsv"
LOG_FAILED="${FOLDER_OUTPUT}/preprocessing-failed.tsv"

# counters
n_total=0
n_success=0
n_failed=0

# write headers
echo -e "participant_id\tsession_id\tscan_age" > "$LOG_SUCCESS"
echo -e "participant_id\tsession_id\tscan_age" > "$LOG_FAILED"

# -----------------------------
# Read TSV (NO PIPE → no subshell)
# -----------------------------
{
    read header  # skip header

    while IFS=$'\t' read -r participant_id session_id scan_age
    do
        ((n_total++))

        SUBJ="sub-${participant_id}_ses-${session_id}"
        OUTDIR="${FOLDER_OUTPUT}/${SUBJ}"

        echo "----------------------------------------"
        echo "Processing ${SUBJ}"

        # -----------------------------
        # Run script
        # -----------------------------
        if bash "$SCRIPT" "$SUBJ" "$FOLDER_INPUT" "$FOLDER_OUTPUT"; then
            echo "[+] Script ran"
        else
            echo "[-] Script failed"
            echo -e "${participant_id}\t${session_id}\t${scan_age}" >> "$LOG_FAILED"
            ((n_failed++))
            continue
        fi

        # -----------------------------
        # Check outputs
        # -----------------------------
        missing=0

        required_files=(
            "${OUTDIR}/${SUBJ}_affine_0GenericAffine.mat"
            "${OUTDIR}/${SUBJ}_T2w_affine.nii.gz"
            "${OUTDIR}/${SUBJ}_LAB_brain_affine.nii.gz"
            "${OUTDIR}/${SUBJ}_LAB43_brain_affine.nii.gz"
        )

        for f in "${required_files[@]}"; do
            if [[ ! -f "$f" ]]; then
                echo "[-] Missing: $f"
                missing=1
            fi
        done

        if [[ "$missing" -eq 0 ]]; then
            echo "[+] SUCCESS"
            echo -e "${participant_id}\t${session_id}\t${scan_age}" >> "$LOG_SUCCESS"
            echo
            ((n_success++))
        else
            echo "[-] FAILED"
            echo -e "${participant_id}\t${session_id}\t${scan_age}" >> "$LOG_FAILED"
            echo
            ((n_failed++))
        fi

    done
} < "$TSV"

# -----------------------------
# Final summary
# -----------------------------
echo "========================================"
echo "Total subjects : $n_total"
echo "Successful     : $n_success"
echo "Failed         : $n_failed"
echo "========================================"
