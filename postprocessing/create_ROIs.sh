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

# Label paths
LABEL_DESCRIPTION="templates/multi-bounti-label-description.txt"

# Input surfaces & metrics
INPUT_SURFACE_MID="${SUBJ_DIR}/${SUBJ}_pred_${HEMI}_midthickness.surf.gii"
INPUT_SURFACE_WM="${SUBJ_DIR}/${SUBJ}_pred_${HEMI}_white.surf.gii"
INPUT_SURFACE_PIAL="${SUBJ_DIR}/${SUBJ}_pred_${HEMI}_pial.surf.gii"
INPUT_METRIC_THICKNESS="${SUBJ_DIR}/${SUBJ}_pred_${HEMI}_thickness.shape.gii"

# Input volume label
INPUT_BOUNTI_LABEL="${SUBJ_DIR}/${SUBJ}_LAB43_brain.nii.gz"

# Output files
OUTPUT_BINARY_MASK="${SUBJ_DIR}/${SUBJ}_pred_${HEMI}_roi.shape.gii"
OUTPUT_CORTICAL_LABELS="${SUBJ_DIR}/${SUBJ}_pred_${HEMI}_rois.label.gii"

# Temporary files
INPUT_BOUNTI_LABEL_FORWB="${SUBJ_DIR}/${SUBJ}_LAB43_brain_temp4wb.nii.gz"
OUTPUT_TMP_CORTICAL_MAP="${SUBJ_DIR}/${SUBJ}_pred_${HEMI}_cleaned_temp.func.gii"

# -------------- Skip if target parcellation files already exist
if [[ -f "$OUTPUT_CORTICAL_LABELS" ]] && [[ -f "$OUTPUT_BINARY_MASK" ]]; then
    echo "[+] Cortical parcellation outputs already exist for ${SUBJ} (${HEMI}). Skipping."
    exit 0
fi

# -------------- Input File Verification
missing_inputs=0
for f in "$INPUT_SURFACE_MID" "$INPUT_SURFACE_WM" "$INPUT_SURFACE_PIAL" "$INPUT_METRIC_THICKNESS" "$INPUT_BOUNTI_LABEL" "$LABEL_DESCRIPTION"; do
    if [[ ! -f "$f" ]]; then
        echo "[-] Error: Missing required input file: $f" >&2
        missing_inputs=1
    fi
done

if [[ "$missing_inputs" -ne 0 ]]; then
    exit 1
fi

echo "[*] Processing cortical parcellation for ${SUBJ} (${HEMI})..."

# -------------- Connectome Workbench Pipeline
# 1. Convert NIfTI volume label for Workbench
wb_command -volume-label-import \
            "${INPUT_BOUNTI_LABEL}" \
            "${LABEL_DESCRIPTION}" \
            "${INPUT_BOUNTI_LABEL_FORWB}"

# 2. Map volumetric BOUNTI labels to midthickness surface constrained by WM and Pial
wb_command -volume-label-to-surface-mapping \
            "${INPUT_BOUNTI_LABEL_FORWB}" \
            "${INPUT_SURFACE_MID}" \
            "${OUTPUT_CORTICAL_LABELS}" \
            -ribbon-constrained \
            "${INPUT_SURFACE_WM}" \
            "${INPUT_SURFACE_PIAL}"

# 3. Create binary ROI mask based on hemisphere-specific cGM labels and non-zero thickness
if [[ "${HEMI}" == "left" ]]; then
    # Left cGM labels: 2, 4, 6, 8, 10, 12
    wb_command -metric-math "((labels == 2) || (labels == 4) || (labels == 6) || (labels == 8) || (labels == 10) || (labels == 12)) && (thickness != 0)" \
                "${OUTPUT_BINARY_MASK}" \
                -var labels    "${OUTPUT_CORTICAL_LABELS}" \
                -var thickness "${INPUT_METRIC_THICKNESS}"
else
    # Right cGM labels: 1, 3, 5, 7, 9, 11
    wb_command -metric-math "((labels == 1) || (labels == 3) || (labels == 5) || (labels == 7) || (labels == 9) || (labels == 11)) && (thickness != 0)" \
                "${OUTPUT_BINARY_MASK}" \
                -var labels    "${OUTPUT_CORTICAL_LABELS}" \
                -var thickness "${INPUT_METRIC_THICKNESS}"
fi

# 4. Mask cortical labels
wb_command -label-mask \
            "${OUTPUT_CORTICAL_LABELS}" \
            "${OUTPUT_BINARY_MASK}" \
            "${OUTPUT_CORTICAL_LABELS}"
# Multiplying by the binary mask forces any vertices outside the ROI to 0
wb_command -metric-math "labels * mask" \
            "${OUTPUT_TMP_CORTICAL_MAP}" \
            -var labels "${OUTPUT_CORTICAL_LABELS}" \
            -var mask "${OUTPUT_BINARY_MASK}"
# Restore the original label names from the text file
wb_command -metric-label-import \
            "${OUTPUT_TMP_CORTICAL_MAP}" \
            "${LABEL_DESCRIPTION}" \
            "${OUTPUT_CORTICAL_LABELS}"

# 5. Clean up binary ROI topology
wb_command -metric-fill-holes "${INPUT_SURFACE_MID}" "${OUTPUT_BINARY_MASK}" "${OUTPUT_BINARY_MASK}"
wb_command -metric-remove-islands "${INPUT_SURFACE_MID}" "${OUTPUT_BINARY_MASK}" "${OUTPUT_BINARY_MASK}"
wb_command -set-map-names "${OUTPUT_BINARY_MASK}" -map 1 "${HEMI}_ROI"

# -------------- Cleanup Temporary Files
rm -f "${OUTPUT_TMP_CORTICAL_MAP}" "${INPUT_BOUNTI_LABEL_FORWB}"

# -------------- Final Output Verification Check
missing_outputs=0

if [[ ! -f "$OUTPUT_CORTICAL_LABELS" ]]; then
    echo "[-] Error: Expected cortical labels output missing: ${OUTPUT_CORTICAL_LABELS}" >&2
    missing_outputs=1
fi

if [[ ! -f "$OUTPUT_BINARY_MASK" ]]; then
    echo "[-] Error: Expected binary mask output missing: ${OUTPUT_BINARY_MASK}" >&2
    missing_outputs=1
fi

if [[ "$missing_outputs" -ne 0 ]]; then
    echo "[-] Error: Cortical parcellation outputs were not successfully created." >&2
    exit 1
fi

echo "[+] Cortical parcellation complete for ${SUBJ} (${HEMI})."
exit 0