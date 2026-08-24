# FetalSeg-CoTAN

## Overview
FetalSeg-CoTAN is a deep learning framework that reconstructs fetal white matter and pial cortical surfaces directly from tissue segmentation labels.

## Repository Status
This repository is currently under active development and is not yet ready for public use.
The codebase, documentation, and setup instructions are being finalized.
Please check back soon for updates. Thank you for your patience!

## Author
Irina Grigorescu

## Example usage

### Preprocessing your data
Assuming your data is in ```MAIN_DATA_FOLDER/```

```
MAIN_DATA_FOLDER/
├── fetal-subjects.tsv
└── input-orig/
    ├── [SUBJ1]
        ├── [SUBJ1]_LAB43_brain.nii.gz
        └── [SUBJ1]_T2w.nii.gz
    └── [SUBJ2]
        ├── [SUBJ2]_LAB43_brain.nii.gz
        └── [SUBJ2]_T2w.nii.gz
```
where: ```fetal-subjects.tsv``` has the following header:
```
participant_id  session_id      scan_age
CC00001XX01       1000          25.0
CC00001XX02       3000          32.0
```
such that ```[SUBJ1]=sub-CC00001XX01_ses-1000``` and ```[SUBJ2]=sub-CC00001XX02_ses-3000```.

Go to ```preprocessing/``` and run the following command:
```
bash multi_preprocess_subject.sh PATH/TO/FOLDER_INPUT PATH/TO/FOLDER_OUTPUT
```
This will do the following:

1) Create simplified multi-bounti labels and brain mask
2) Register original T2w image to T2w template and apply the transformation to the labels

Your folder will now look like this:
```
MAIN_DATA_FOLDER/
├── fetal-subjects.tsv
├── input-orig/
    ├── [SUBJ1]
        ├── [SUBJ1]_brain-mask.nii.gz           ## this is new (brain mask)
        ├── [SUBJ1]_LAB43_brain.nii.gz
        ├── [SUBJ1]_LAB_brain.nii.gz            ## this is new (simplified labels)
        └── [SUBJ1]_T2w.nii.gz
    └── [SUBJ2] ...
└── input-aff/                                  ## this is new (the output folder)
    ├── preprocessing-failed.tsv                ## Logs failed cases
    ├── preprocessing-success.tsv               ## Logs succesful cases
    ├── [SUBJ1]
        ├── [SUBJ1]_affine_0GenericAffine.mat   ## this is new (the affine matrix)
        ├── [SUBJ1]_LAB43_brain_affine.nii.gz   ## this is new (the affinely aligned multi bounti labels)
        ├── [SUBJ1]_LAB_brain_affine.nii.gz     ## this is new (the affinely aligned simplified labels)
        └── [SUBJ1]_T2w_affine.nii.gz           ## this is new (the affinely aligned T2w image)
    └── [SUBJ2] ...   
```

### Running inference
Assuming your data is in ```MAIN_DATA_FOLDER/input-aff/``` as shown above,
navigate to the **FetalSeg-CoTAN root directory** (or project root) and run the following command:
```
python -m predict_all \
          --tsv_file_subjects="/PATH/TO/fetal-subjects.tsv" \
          --results_file="fetal-metrics.csv" \
          --logs_file="fetal-logs.csv" \
          --templates_path="templates/" \
          --orig_t2w_path="MAIN_DATA_FOLDER/input-orig/" \
          --orig_lab_path="MAIN_DATA_FOLDER/input-orig/" \
          --affine_label_path="MAIN_DATA_FOLDER/input-aff/" \
          --output_path="MAIN_DATA_FOLDER/output/" \
          --device="cuda" \
          --do_spheres --do_rois --keep_intermediates 
```
The last three parameters ```--do_spheres --do_rois --keep_intermediates``` 
you can omit if you want only surfaces without spheres and without the cortical parcellation.

After a succesful run, your folder will look like this:
```
MAIN_DATA_FOLDER/
├── fetal-subjects.tsv
├── input-orig/
    ├── [SUBJ1]
        ├── ...
        └── [SUBJ1]_T2w.nii.gz
    └── ...
├── input-aff/                                  
    ├── preprocessing-failed.tsv           
    ├── preprocessing-success.tsv            
    ├── [SUBJ1]
        ├── ...
        ├── [SUBJ1]_LAB_brain_affine.nii.gz
        └── [SUBJ1]_T2w_affine.nii.gz
    └── ...   
└── output/                                     ## This is new
    ├── fetal-logs.csv                          ## Stores fetal logs
    ├── fetal-metrics.csv                       ## Stores average per hemisphere fetal metrics            
    ├── [SUBJ1]        
        ├── [SUBJ1]_pred_[left/right]_[white/pial/midthickness[.surf.gii    ## surfaces for each hemisphere
        ├── [SUBJ1]_pred_[left/right]_[curv/sulc/thickness].shape.gii       ## metrics for each hemisphere
        ├── [SUBJ1]_pred_[left/right]_[sphere/inflated/vinflated].surf.gii  ## inflated surfaces for each hemisphere (if --do_sphere set)
        ├── [SUBJ1]_pred_[left/right]_roi.shape.gii                         ## cortical binary  mask (if --do_rois set)
        ├── [SUBJ1]_pred_[left/right]_rois.label.gii                        ## cortical parcellation (if --do_rois set)
        ├── [SUBJ1]_pred_[left/right]_[rois_filled/rois_cleaned].label.gii  ## cleaned-up cortical parcellations (if --keep_intermediates set, otherwise overwrites rois.label.gii)
        ├── [SUBJ1]_LAB43_brain.nii.gz                                      ## the original/native multi-bounti labels
        └── [SUBJ1]_T2w.nii.gz                                              ## the original/native T2w image
    └── [SUBJ2]
        └── ...
```

## Related Repositories

### Associated Repository
The following repository is part of my work for MIDL 2026 if you would like to check it out:
1. SuD-CoTAN - [https://github.com/irinagrigorescu/SuDCoTAN](https://github.com/irinagrigorescu/SuDCoTAN)

### Acknowledgements
The following repositories were not only of great inspiration, but have also helped make this work possible.
Please do check them out:
1. CoTAN - [https://github.com/m-qiang/CoTAN](https://github.com/m-qiang/CoTAN)
2. CoSEG - [https://github.com/m-qiang/CoSeg](https://github.com/m-qiang/CoSeg)
