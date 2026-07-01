########################################################################
###### IRINA GRIGORESCU
######
###### This file contains the input/output code needed for this repository
########################################################################

import ants
import nibabel as nib
import os
import numpy as np
from nibabel.gifti import gifti


def load_T2w_template_and_affine(fpath):
    """
    Load the dhcp_fetal_week36_t2w.nii.gz and its affine matrix

    :param fpath: path to the location of the template T2w data
    :return: ants image and affine matrix
    """
    # Load data
    img_t2_atlas_ants = ants.image_read(os.path.join(fpath, "dhcp_fetal_week36_t2w.nii.gz"))
    img_t2_atlas_nib  = nib.load(os.path.join(fpath, "dhcp_fetal_week36_t2w.nii.gz"))
    affine_mat = img_t2_atlas_nib.affine

    return img_t2_atlas_ants, affine_mat


def load_initial_surfaces(fpath, hemi):
    """
    Load the initial surfaces used as input to the network

    :param fpath: path to the location of the initial surfaces
    :param hemi: left/right
    :return:
    """
    # Load surface initial
    surf_in = nib.load(os.path.join(fpath, f"week36_{hemi}_vinflated_symmetry_affine.ico6.surf.gii"))

    return surf_in


def check_file_exists(path, description):
    """
    Check if file exists, print status, and exit if missing

    :param path: path to location of file
    :param description: description of file
    :return: 0 if ok / -1 if does not exist
    """
    if os.path.exists(path):
        print(f"[ OK  ] {description} file  exists: {path}")
        return 0
    else:
        print(f"[ERROR] {description} file missing: {path}\nExiting...")
        return -1


###################################################
################################################### Please check individual repositories for original code
###################################################
def save_gifti_surface(vert, face, save_dir, surf_hemi='left', surf_type='wm'):
    """
    Save gifti surface (.surf.gii).

    For original code please see: https://github.com/m-qiang/dhcp-dl-neonatal/blob/main/utils/io.py

    Inputs:
    :param vert: mesh vertices, (|V|,3) numpy.float32
    :param face: mesh faces, (|F|,3) numpy.int32
    :param save_dir: directory for saving, string
    :param surf_hemi: ['left', 'right']
    :param surf_type: ['wm', 'pial', 'midthickness',
                       'inflated', 'vinflated', 'sphere']
    """

    # convert args to gifti header
    if surf_hemi == 'left':
        _surf_hemi = 'CortexLeft'
    elif surf_hemi == 'right':
        _surf_hemi = 'CortexRight'

    if surf_type == 'wm':
        _surf_type = 'GrayWhite'
        _geo_type = 'Anatomical'
    elif surf_type == 'pial':
        _surf_type = 'Pial'
        _geo_type = 'Anatomical'
    elif surf_type == 'midthickness':
        _surf_type = 'MidThickness'
        _geo_type = 'Anatomical'
    elif surf_type == 'inflated':
        _surf_type = 'MidThickness'
        _geo_type = 'Inflated'
    elif surf_type == 'vinflated':
        _surf_type = 'MidThickness'
        _geo_type = 'VeryInflated'
    elif surf_type == 'sphere':
        _surf_type = 'MidThickness'
        _geo_type = 'Spherical'

    # meta data
    vert_meta_dict = {'<![CDATA[AnatomicalStructurePrimary]]>':
                          '<![CDATA[' + _surf_hemi + ']]>',
                      '<![CDATA[AnatomicalStructureSecondary]]>':
                          '<![CDATA[' + _surf_type + ']]>',
                      '<![CDATA[GeometricType]]>':
                          '<![CDATA[' + _geo_type + ']]>',
                      '<![CDATA[Name]]>': '<![CDATA[#1]]>'}
    face_meta_dict = {'<![CDATA[Name]]>': '<![CDATA[#2]]>'}

    vert_meta = gifti.GiftiMetaData(vert_meta_dict)
    face_meta = gifti.GiftiMetaData(face_meta_dict)

    # create gifti data
    gii_surf = gifti.GiftiImage()
    gii_surf_vert = gifti.GiftiDataArray(
        vert.astype(np.float32), intent='pointset', meta=vert_meta)
    gii_surf_face = gifti.GiftiDataArray(
        face.astype(np.int32), intent='triangle', meta=face_meta)
    gii_surf.add_gifti_data_array(gii_surf_vert)
    gii_surf.add_gifti_data_array(gii_surf_face)

    # save gifti xml file (.gii)
    gii_file = gii_surf.to_xml().decode("utf-8");
    gii_file = gii_file.replace("&lt;", "<");
    gii_file = gii_file.replace("&gt;", ">");

    with open(save_dir, 'wb') as f:
        f.write(gii_file.encode("utf-8"))
    # nib.save(gii_surf, save_dir)


def save_gifti_metric(metric, save_dir, surf_hemi='left', metric_type='curv'):
    """
    Save gifti metric (.shape.gii).

    For original code please see: https://github.com/m-qiang/dhcp-dl-neonatal/blob/main/utils/io.py

    :param metric: mesh metric, (|V|) numpy.float32
    :param save_dir: directory for saving, string
    :param surf_hemi: ['left', 'right']
    :param metric_type: ['thickness', 'curv', 'sulc']
    """

    # convert args to gifti header
    if surf_hemi == 'left':
        _surf_hemi = 'CortexLeft'
    elif surf_hemi == 'right':
        _surf_hemi = 'CortexRight'

    # set meta data
    if metric_type == 'thickness':
        _metric_type = 'Thickness'
        ScaleMode = 'MODE_AUTO_SCALE_PERCENTAGE'
        AutoScalePercentageValues = '98.000000 2.000000 7.000000 98.000000'
        UserScaleValues = '-100.000000 0.000000 0.000000 100.000000'
        PaletteName = 'videen_style'
        DisplayPositiveData = 'true'
        DisplayZeroData = 'false'
        DisplayNegativeData = 'false'

    elif metric_type == 'curv':
        _metric_type = 'Curvature'
        ScaleMode = 'MODE_AUTO_SCALE_PERCENTAGE'
        AutoScalePercentageValues = '98.000000 2.000000 2.000000 98.000000'
        UserScaleValues = '-100.000000 0.000000 0.000000 100.000000'
        PaletteName = 'PSYCH-NO-NONE'
        DisplayPositiveData = 'true'
        DisplayZeroData = 'true'
        DisplayNegativeData = 'true'

    elif metric_type == 'sulc':
        _metric_type = 'SulcalDepth'
        ScaleMode = 'MODE_AUTO_SCALE_PERCENTAGE'
        AutoScalePercentageValues = '98.000000 2.000000 2.000000 98.000000'
        UserScaleValues = '-100.000000 0.000000 0.000000 100.000000'
        PaletteName = 'ROY-BIG-BL'
        DisplayPositiveData = 'true'
        DisplayZeroData = 'true'
        DisplayNegativeData = 'true'

    # meta data for color map
    cmap_meta = \
        '<PaletteColorMapping Version="1">\n' + \
        '<ScaleMode>' + ScaleMode + '</ScaleMode>\n' + \
        '<AutoScalePercentageValues>' + AutoScalePercentageValues + '</AutoScalePercentageValues>\n' + \
        '<AutoScaleAbsolutePercentageValues>2.000000 98.000000</AutoScaleAbsolutePercentageValues>\n' + \
        '<UserScaleValues>' + UserScaleValues + '</UserScaleValues>\n' + \
        '<PaletteName>' + PaletteName + '</PaletteName>\n' + \
        '<InterpolatePalette>true</InterpolatePalette>\n' + \
        '<DisplayPositiveData>' + DisplayPositiveData + '</DisplayPositiveData>\n' + \
        '<DisplayZeroData>' + DisplayZeroData + '</DisplayZeroData>\n' + \
        '<DisplayNegativeData>' + DisplayNegativeData + '</DisplayNegativeData>\n' + \
        '<ThresholdTest>THRESHOLD_TEST_SHOW_OUTSIDE</ThresholdTest>\n' + \
        '<ThresholdType>THRESHOLD_TYPE_OFF</ThresholdType>\n' + \
        '<ThresholdFailureInGreen>false</ThresholdFailureInGreen>\n' + \
        '<ThresholdNormalValues>-1.000000 1.000000</ThresholdNormalValues>\n' + \
        '<ThresholdMappedValues>-1.000000 1.000000</ThresholdMappedValues>\n' + \
        '<ThresholdMappedAvgAreaValues>-1.000000 1.000000</ThresholdMappedAvgAreaValues>\n' + \
        '<ThresholdDataName></ThresholdDataName>\n' + \
        '<ThresholdRangeMode>PALETTE_THRESHOLD_RANGE_MODE_MAP</ThresholdRangeMode>\n' + \
        '<ThresholdLowHighLinked>false</ThresholdLowHighLinked>\n' + \
        '<NumericFormatMode>AUTO</NumericFormatMode>\n' + \
        '<PrecisionDigits>2</PrecisionDigits>\n' + \
        '<NumericSubivisions>0</NumericSubivisions>\n' + \
        '<ColorBarValuesMode>DATA</ColorBarValuesMode>\n' + \
        '<ShowTickMarksSelected>false</ShowTickMarksSelected>'

    metric_meta_dict = {'<![CDATA[Name]]>': '<![CDATA[' + _metric_type + ']]>',
                        '<![CDATA[PaletteColorMapping]]>':
                            '<![CDATA[' + cmap_meta + ']]>'}
    metric_meta = gifti.GiftiMetaData(metric_meta_dict)

    # meta data
    metric = metric.astype(np.float32)
    gii_meta_dict = {'<![CDATA[AnatomicalStructurePrimary]]>':
                         '<![CDATA[' + _surf_hemi + ']]>'}
    gii_meta = gifti.GiftiMetaData(gii_meta_dict)

    # new gii image
    gii_metric = gifti.GiftiImage(meta=gii_meta)
    gii_metric_arr = gifti.GiftiDataArray(metric, intent='normal', meta=metric_meta)
    gii_metric.add_gifti_data_array(gii_metric_arr)

    # save gifti xml file (.gii)
    gii_file = gii_metric.to_xml().decode("utf-8");
    gii_file = gii_file.replace("&lt;", "<");
    gii_file = gii_file.replace("&gt;", ">");

    with open(save_dir, 'wb') as f:
        f.write(gii_file.encode("utf-8"))
