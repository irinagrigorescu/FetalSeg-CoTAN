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
        ├── [SUBJ1]_brain-mask.nii.gz           ## this is new
        ├── [SUBJ1]_LAB43_brain.nii.gz
        ├── [SUBJ1]_LAB_brain.nii.gz            ## this is new
        └── [SUBJ1]_T2w.nii.gz
    └── [SUBJ2] ...
└── input-aff/                                  ## this is new
    ├── preprocessing-failed.tsv                ## Logs failed cases
    ├── preprocessing-success.tsv               ## Logs succesful cases
    ├── [SUBJ1]
        ├── [SUBJ1]_affine_0GenericAffine.mat
        ├── [SUBJ1]_LAB43_brain_affine.nii.gz
        ├── [SUBJ1]_LAB_brain_affine.nii.gz
        └── [SUBJ1]_T2w_affine.nii.gz
    └── [SUBJ2] ...   
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
