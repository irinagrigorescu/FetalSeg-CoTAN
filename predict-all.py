########################################################################
###### IRINA GRIGORESCU
######
###### This script predict all surfaces needed for fetal DHCP
########################################################################

# Input:  Lab affine subject (input to network)
#         Initial surfaces as input to the network
# -- input should also be a tsv file where:
#    participant_id  session_id      scan_age        dataset
#    CC00856XX15     3530    37.43   test
#    CC00888XX23     2732    35.29   train
#    CC01019XX13     46330   21.14   FAIL

# for any subject you need to
# have the data pre-affinely aligned to a template
#
# 1. Load T2w template / Label affine (input to network) / T2w original
# 2. Predict left/right wm/pial
# 3. Save surfaces in the original space
# 4. Run freesurfer with subprocess to inflate surface and obtain:
#    a) inflated surface
#    b) very-inflated surface
#    c) subject specific sphere
#    d) sulcal depth
# 5. Run python code to produce:
#    a) curvature
#    b) cortical thickness
#    c) surface area


# Output: Surfaces predicted should be saved in the original space +
#         Sulcal Depth +
#         Curvature +
#         Cortical Thickness +
#         Subject Sphere (Freesurfer) +
#         Inflated +
#         V-Inflated
####################
import os
import torch
import time
import numpy as np
import pandas as pd
import argparse
import nibabel as nib

from src.io import (
    load_T2w_template_and_affine, load_initial_surfaces,
    check_file_exists,
    save_gifti_surface, save_gifti_metric
)
from src.networks import (
    CoTAN, CoTANPial
)
from src.processing import (
    preprocess_surfaces,
    postprocess_surfaces
)
from src.mesh import (
    taubin_smooth, face_area
)
from src.metrics import (
    sulcal_depth, calculate_average_thickness, curvature, metric_dilation
)


# ========== MAIN FUNCTION  ==========
def run_predict_all(args):
    """
    This runs the compute cortical thickness, curvature and sulcal depth
        for all subjects in a given dataset
    :param args:
    :return:
    """
    # -------------- LOAD ARGUMENTS
    tsv_file_subjects = args.tsv_file_subjects  # path to tsv file with all subjects
    templates_path    = args.templates_path  # path to initial surfaces templates and t2w template
    affine_label_path = args.affine_label_path  # the path to the affinely registered T2w
    output_path       = args.output_path  # the path to the outputs
    device            = args.device  # whether cpu or cuda
    results_file      = args.results_file  # file where all metrics are stored

    # -------------- SET ARGUMENTS
    step_size = 0.02
    do_left, do_right = True, True

    # -------------- INITIALISE MODELS
    print('Initalize models ...')
    model_left_white  = CoTAN(    layers=[16, 32, 64, 128, 128], M=4, R=3, device=device).to(device)
    model_left_pial   = CoTANPial(layers=[16, 32, 32, 32, 32],   M=4, R=3, device=device).to(device)
    model_right_white = CoTAN(    layers=[16, 32, 64, 128, 128], M=4, R=3, device=device).to(device)
    model_right_pial  = CoTANPial(layers=[16, 32, 32, 32, 32],   M=4, R=3, device=device).to(device)

    ############################# BELOW are
    ############################# the MB models which were trained on bounti-43 for
    ############################# 1) 600E for WM on old surfaces + 50E on new surfaces with extra iterations
    ############################# 2) 800E for pial on old surfaces + 50E on new surfaces with extra iterations
    ############################# and saved in output-data-COTAN-wLAB-dhcp-MB-MBv4/

    # -------------- LOAD PRE-TRAINED MODELS
    model_path_white = "./model-weights/"
    model_path_pial  = "./model-weights/"

    # White Matter
    if do_left:
        model_left_white_name    = "model_dhcp_fetal_multibounti_wLabels_left_white.pt"
    if do_right:
        model_right_white_name   = "model_dhcp_fetal_multibounti_wLabels_right_white.pt"
    # Pial
    if do_left:
        model_left_pial_name   = "model_dhcp_fetal_multibounti_wLabels_left_pial.pt"
    if do_right:
        model_right_pial_name  = "model_dhcp_fetal_multibounti_wLabels_right_pial.pt"

    if do_left:
        model_left_white.load_state_dict(torch.load(model_path_white + model_left_white_name, map_location=device))
        model_left_pial.load_state_dict(torch.load(model_path_pial + model_left_pial_name, map_location=device))
    if do_right:
        model_right_white.load_state_dict(torch.load(model_path_white + model_right_white_name, map_location=device))
        model_right_pial.load_state_dict(torch.load(model_path_pial + model_right_pial_name, map_location=device))

    # -------------- LOAD TEMPLATE AFFINE
    print('Load T2w template ...')
    t2w_template_ants, t2w_template_affine = load_T2w_template_and_affine(fpath=templates_path)

    # -------------- LOAD INITIAL SURFACES
    init_surfaces = {x : None for x in ["36w-left", "36w-right"]}
    if do_left:
        surf_left_in = load_initial_surfaces(templates_path, "left")
        v_left_in, f_left_in = preprocess_surfaces(surf_left_in, "left", t2w_template_affine, flag="init")
        init_surfaces["36w-left"] = (v_left_in, f_left_in)
    if do_right:
        surf_right_in = load_initial_surfaces(templates_path, "right")
        v_right_in, f_right_in = preprocess_surfaces(surf_right_in, "right", t2w_template_affine, flag="init")
        init_surfaces["36w-right"] = (v_right_in, f_right_in)

    # -------------- INPUT INTEGRATION TIME SEQUENCE
    input_time_sequence = torch.arange(1. / step_size).to(device).unsqueeze(1) * step_size

    # -------------- READ IN TSV FILE
    subjects_pd = pd.read_csv(tsv_file_subjects, sep="\t", dtype="str").to_dict("list")

    # check all keys exist
    if "participant_id" not in subjects_pd.keys() or \
            "session_id" not in subjects_pd.keys() or \
            "scan_age" not in subjects_pd.keys():
        print(f"[ERROR] No [participant_id, session_id, scan_age] keys in the tsv file {tsv_file_subjects}")
        print("Exiting...")
        return -1

    # get names/ages and data location
    subjects_names    = ["sub-" + sub_ + "_ses-" + str(ses_) for sub_, ses_ in zip(subjects_pd['participant_id'],
                                                                                subjects_pd['session_id'])]
    subjects_ages     = [age_ for age_ in subjects_pd['scan_age']]
    n_subjects        = len(subjects_names)

    ###############################################################
    ###############################################################
    # -------------- SAVE METRICS IN HERE
    all_keys = ["subj", "scan_age",  # info about the subject
                "mean_cthck_L", "mean_cthck_R", "std_cthck_L", "std_cthck_R",             # cortical thickness
                "mean_curv_L", "mean_curv_R", "std_curv_L", "std_curv_R",                 # curvature
                "mean_poscurv_L", "mean_poscurv_R", "mean_negcurv_L", "mean_negcurv_R",
                "mean_sulc_L", "mean_sulc_R", "std_sulc_L", "std_sulc_R",                 # sulc depth
                "mean_possulc_L", "mean_possulc_R", "mean_negsulc_L", "mean_negsulc_R",
                "surf_area_L", "surf_area_R", "mean_surf_area_L", "mean_surf_area_R"]     # surface area
    csv_results_data = {k: ["NA"] * len(subjects_names) for k in all_keys}

    ###############################################################
    ###############################################################
    # -------------- GO THROUGH ALL THE SUBJECTS
    for i_s, curr_subj in enumerate(subjects_names):
        # TODO: run for one subject only first
        # if i_s > 2: break
        # if curr_loc != "test": continue

        # -------------- Get current subject's age
        curr_age = float(subjects_ages[i_s])
        print(f"{i_s + 1:3d} / {n_subjects:3d} | {curr_subj:28s} {curr_age:5.2f}")
        # ---> save subject name and age
        csv_results_data["subj"][i_s] = curr_subj
        csv_results_data["scan_age"][i_s] = curr_age

        # if curr_subj != "sub-CC00974XX18_ses-34631": continue
        # if curr_subj != "sub-CC01098XX19_ses-78530": continue
        # if curr_age < 33.0 : continue

        # ---------------------------------- CREATE SUBFOLDER
        final_path = os.path.join(output_path, curr_subj)
        os.makedirs(final_path, exist_ok=True)

        # -------------- Get current subject's path to the affinely aligned label data
        curr_aff_lab = f"{affine_label_path}/{curr_subj}/{curr_subj}_LAB_brain_affine.nii.gz"

        # Check the paths exist
        if check_file_exists(curr_aff_lab, "affine label") == -1: return -1

        # -------------- LOAD data and affines
        print('\nLoad Label data ...')
        curr_subj_volume = nib.load(curr_aff_lab).get_fdata()
        affine_t2_align  = nib.load(curr_aff_lab).affine

        # -------------- INPUT VOLUME AS TENSOR AND SPLIT BETWEEN LEFT AND RIGHT
        if do_left:
            curr_subj_volume_left = torch.Tensor(curr_subj_volume[None, None]).to(device)[:, :, 88:]
        if do_right:
            curr_subj_volume_right = torch.Tensor(curr_subj_volume[None, None]).to(device)[:, :, :88]

        # -------------- INPUT AGE
        input_age = (curr_age - 20) / 20.  # normalize age
        input_age = torch.Tensor(np.array(input_age)[None]).to(device)
        input_age = input_age.repeat(int(1. / step_size), 1).to(device)

        ###############################################################
        ###############################################################
        # ---------------------------------- RUN INFERENCE
        ########################
        print('\nStart surface reconstruction ...')
        t_start_start = time.time()
        t_start = time.time()
        # Get the current weekly template for left and right
        if do_left:
            (curr_v_left_in, curr_f_left_in)   = init_surfaces["36w-left"]
        if do_right:
            (curr_v_right_in, curr_f_right_in) = init_surfaces["36w-right"]
        # Send them to gpu
        if do_left:
            curr_v_left_in, curr_f_left_in = curr_v_left_in.to(device), curr_f_left_in.to(device)
        if do_right:
            curr_v_right_in, curr_f_right_in = curr_v_right_in.to(device), curr_f_right_in.to(device)
        with torch.no_grad():
            if do_left:
                ## WM - LEFT
                v_left_white = model_left_white(curr_v_left_in, input_time_sequence, input_age, curr_subj_volume_left)
                ## WM - LEFT - through taubin
                v_left_white = taubin_smooth(v_left_white, curr_f_left_in, n_iters=10)
                ## PIAL - LEFT
                v_left_pial = model_left_pial(v_left_white, input_time_sequence, input_age, curr_subj_volume_left)

            if do_right:
                ## WM - RIGHT
                v_right_white = model_right_white(curr_v_right_in, input_time_sequence, input_age, curr_subj_volume_right)
                ## WM - RIGHT - through taubin
                v_right_white = taubin_smooth(v_right_white, curr_f_right_in, n_iters=10)
                ## PIAL - RIGHT
                v_right_pial = model_right_pial(v_right_white, input_time_sequence, input_age, curr_subj_volume_right)
        t_end = time.time()
        print(f"Finished. Runtime:{np.round(t_end - t_start, 4)}")

        # ---------------------------------- POST PROCESS THE PREDICTIONS
        ########################
        if do_left:
            (vert_left_white_final, vert_left_pial_final,
             faces_left_final) = postprocess_surfaces(v_left_white, v_left_pial,
                                                      curr_f_left_in, "left", affine_t2_align)
        if do_right:
            (vert_right_white_final, vert_right_pial_final,
             faces_right_final) = postprocess_surfaces(v_right_white, v_right_pial,
                                                       curr_f_right_in, "right", affine_t2_align)

        # ---------------------------------- OTHER SURFACES - midthickness
        ########################
        if do_left:
            vert_left_mid_final = (vert_left_white_final + vert_left_pial_final) / 2.0
        if do_right:
            vert_right_mid_final = (vert_right_white_final + vert_right_pial_final) / 2.0

        ###############################################################
        ###############################################################
        # ---------------------------------- METRICS ON THE SURFACE
        ########################
        print('\nStart sulcal depth calculation for left and right ...')
        t_start = time.time()
        if do_left:
            _, curr_left_sulc = sulcal_depth(torch.Tensor(vert_left_white_final).unsqueeze(0).to(device),
                                             torch.LongTensor(faces_left_final).unsqueeze(0).to(device),
                                             return_vertices=True, verbose=False)
        if do_right:
            _, curr_right_sulc = sulcal_depth(torch.Tensor(vert_right_white_final).unsqueeze(0).to(device),
                                              torch.LongTensor(faces_right_final).unsqueeze(0).to(device),
                                              return_vertices=True, verbose=False)
        t_end = time.time()
        print(f"Finished. Runtime:{np.round(t_end - t_start, 4)}")

        ########################
        print('\nStart cortical thickness calculation for left and right ...')
        t_start = time.time()
        if do_left:
            ct_mean_L, ct_std_L, _, _, _, _, _, cthickness_map_left = calculate_average_thickness(vert_left_white_final,
                                                                                                  vert_left_pial_final,
                                                                                                  return_surface=True)
            curr_left_thickness = metric_dilation(torch.Tensor(cthickness_map_left[None, :, None]).to(device),
                                                  torch.LongTensor(faces_left_final).unsqueeze(0).to(device),
                                                  n_iters=10)
        if do_right:
            ct_mean_R, ct_std_R, _, _, _, _, _, cthickness_map_right = calculate_average_thickness(vert_right_white_final,
                                                                                                   vert_right_pial_final,
                                                                                                   return_surface=True)
            curr_right_thickness = metric_dilation(torch.Tensor(cthickness_map_right[None, :, None]).to(device),
                                                   torch.LongTensor(faces_right_final).unsqueeze(0).to(device),
                                                   n_iters=10)
        t_end = time.time()
        print(f"Finished. Runtime:{np.round(t_end - t_start, 4)}")

        ########################
        print('\nStart curvature calculation for left and right ...')
        t_start = time.time()
        if do_left:
            curr_left_curv = curvature(torch.Tensor(vert_left_white_final).unsqueeze(0).to(device),
                                       torch.LongTensor(faces_left_final).unsqueeze(0).to(device),
                                       smooth_iters=5)
        if do_right:
            curr_right_curv = curvature(torch.Tensor(vert_right_white_final).unsqueeze(0).to(device),
                                        torch.LongTensor(faces_right_final).unsqueeze(0).to(device),
                                        smooth_iters=5)
        t_end = time.time()
        print(f"Finished. Runtime:{np.round(t_end - t_start, 4)}")

        ########################
        print('\nStart face area calculation for left and right ...')
        t_start = time.time()
        if do_left:
            curr_left_farea = face_area(torch.Tensor(vert_left_white_final).unsqueeze(0).to(device),
                                        torch.LongTensor(faces_left_final).unsqueeze(0).to(device))
        if do_right:
            curr_right_farea = face_area(torch.Tensor(vert_right_white_final).unsqueeze(0).to(device),
                                         torch.LongTensor(faces_right_final).unsqueeze(0).to(device))
        t_end = time.time()
        print(f"Finished. Runtime:{np.round(t_end - t_start, 4)}\n")

        ###############################################################
        ###############################################################
        # ---------------------------------- METRICS ON THE SURFACE - scalar valued data
        # ---------------------------------- Store Cortical Thickness - LEFT
        if do_left:
            csv_results_data["mean_cthck_L"][i_s] = ct_mean_L
            csv_results_data["std_cthck_L"][i_s]  = ct_std_L
        # ---------------------------------- Store Cortical Thickness - RIGHT
        if do_right:
            csv_results_data["mean_cthck_R"][i_s] = ct_mean_R
            csv_results_data["std_cthck_R"][i_s] = ct_std_R

        # ---------------------------------- Store Curvature - LEFT
        if do_left:
            csv_results_data["mean_curv_L"][i_s] = float(np.mean(np.abs(curr_left_curv)))
            csv_results_data["std_curv_L"][i_s] = float(np.std(curr_left_curv))
            csv_results_data["mean_poscurv_L"][i_s] = float(np.mean(curr_left_curv[curr_left_curv > 0.0]))
            csv_results_data["mean_negcurv_L"][i_s] = float(np.mean(curr_left_curv[curr_left_curv < 0.0]))
        # ---------------------------------- Store Curvature - RIGHT
        if do_right:
            csv_results_data["mean_curv_R"][i_s] = float(np.mean(np.abs(curr_right_curv)))
            csv_results_data["std_curv_R"][i_s] = float(np.std(curr_right_curv))
            csv_results_data["mean_poscurv_R"][i_s] = float(np.mean(curr_right_curv[curr_right_curv > 0.0]))
            csv_results_data["mean_negcurv_R"][i_s] = float(np.mean(curr_right_curv[curr_right_curv < 0.0]))

        # ---------------------------------- Store Sulcal Depth - LEFT
        if do_left:
            csv_results_data["mean_sulc_L"][i_s] = float(np.mean(np.abs(curr_left_sulc)))
            csv_results_data["std_sulc_L"][i_s] = float(np.std(curr_left_sulc))
            csv_results_data["mean_possulc_L"][i_s] = float(np.mean(curr_left_sulc[curr_left_sulc > 0.0]))
            csv_results_data["mean_negsulc_L"][i_s] = float(np.mean(curr_left_sulc[curr_left_sulc < 0.0]))
        # ---------------------------------- Store Sulcal Depth - RIGHT
        if do_right:
            csv_results_data["mean_sulc_R"][i_s] = float(np.mean(np.abs(curr_right_sulc)))
            csv_results_data["std_sulc_R"][i_s] = float(np.std(curr_right_sulc))
            csv_results_data["mean_possulc_R"][i_s] = float(np.mean(curr_right_sulc[curr_right_sulc > 0.0]))
            csv_results_data["mean_negsulc_R"][i_s] = float(np.mean(curr_right_sulc[curr_right_sulc < 0.0]))

        # ---------------------------------- Store Surface Area - LEFT
        if do_left:
            csv_results_data["surf_area_L"][i_s] = float(curr_left_farea.sum().item())
            csv_results_data["mean_surf_area_L"][i_s] = float(curr_left_farea.mean().item())
        # ---------------------------------- Store Surface Area - RIGHT
        if do_right:
            csv_results_data["surf_area_R"][i_s] = float(curr_right_farea.sum().item())
            csv_results_data["mean_surf_area_R"][i_s] = float(curr_right_farea.mean().item())
        ###############################################################
        ###############################################################

        ###############################################################
        ###############################################################
        # ---------------------------------- SAVE PREDICTIONS
        print('\nSave surface meshes ...')
        ## SAVE LEFT
        if do_left:
            save_gifti_surface(vert_left_white_final, faces_left_final,
                               save_dir=os.path.join(final_path, f"{curr_subj}_pred_left_white.surf.gii"),
                               surf_hemi="left", surf_type="wm")
            save_gifti_surface(vert_left_pial_final, faces_left_final,
                               save_dir=os.path.join(final_path, f"{curr_subj}_pred_left_pial.surf.gii"),
                               surf_hemi="left", surf_type="pial")
            save_gifti_surface(vert_left_mid_final, faces_left_final,
                               save_dir=os.path.join(final_path, f"{curr_subj}_pred_left_midthickness.surf.gii"),
                               surf_hemi="left", surf_type="midthickness")
            save_gifti_metric(metric=curr_left_sulc,
                              save_dir=os.path.join(final_path, f"{curr_subj}_pred_left_sulc.shape.gii"),
                              surf_hemi="left", metric_type="sulc")
            save_gifti_metric(metric=curr_left_curv,
                              save_dir=os.path.join(final_path, f"{curr_subj}_pred_left_curv.shape.gii"),
                              surf_hemi="left", metric_type="curv")
            save_gifti_metric(metric=curr_left_thickness,
                              save_dir=os.path.join(final_path, f"{curr_subj}_pred_left_thickness.shape.gii"),
                              surf_hemi="left", metric_type="thickness")

        ## SAVE RIGHT
        if do_right:
            save_gifti_surface(vert_right_white_final, faces_right_final,
                               save_dir=os.path.join(final_path, f"{curr_subj}_pred_right_white.surf.gii"),
                               surf_hemi="right", surf_type="wm")
            save_gifti_surface(vert_right_pial_final, faces_right_final,
                               save_dir=os.path.join(final_path, f"{curr_subj}_pred_right_pial.surf.gii"),
                               surf_hemi="right", surf_type="pial")
            save_gifti_surface(vert_right_mid_final, faces_right_final,
                               save_dir=os.path.join(final_path, f"{curr_subj}_pred_right_midthickness.surf.gii"),
                               surf_hemi="right", surf_type="midthickness")
            save_gifti_metric(metric=curr_right_sulc,
                              save_dir=os.path.join(final_path, f"{curr_subj}_pred_right_sulc.shape.gii"),
                              surf_hemi="right", metric_type="sulc")
            save_gifti_metric(metric=curr_right_curv,
                              save_dir=os.path.join(final_path, f"{curr_subj}_pred_right_curv.shape.gii"),
                              surf_hemi="right", metric_type="curv")
            save_gifti_metric(metric=curr_right_thickness,
                              save_dir=os.path.join(final_path, f"{curr_subj}_pred_right_thickness.shape.gii"),
                              surf_hemi="right", metric_type="thickness")

        ###############################################################
        ###############################################################

        t_end_end = time.time()
        print(f"\nDone with subject {curr_subj:28s} {curr_age:5.2f} Runtime: {np.round(t_end_end - t_start_start, 4)} \n\n")

    ###############################################################
    ###############################################################
    # -------------- SAVE THE METRICS
    # convert your dict to a DataFrame
    df = pd.DataFrame(csv_results_data)
    # save to CSV (no index column)
    df.to_csv(os.path.join(output_path, results_file), index=False)


# ========== RUN CALCULATION ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict all fetal dHCP surfaces",
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--tsv_file_subjects',
                        default='fetal-subjects.tsv',
                        type=str,
                        help=r"""path to tsv file with all subjects
                        e.g. participant_id session_id  scan_age
                             CC00001XX01    1000        25.0""")

    parser.add_argument('--results_file',
                        default='fetal-metrics.csv',
                        type=str,
                        help=r"""file where all the metrics calculated here will be stored
                                 location will be at --output_path / --results_file""")

    parser.add_argument('--templates_path',
                        default="templates/",
                        type=str,
                        help=r"""path to initial weekly surfaces and template t2w for alignment
                            (surfaces are used for input to the model and T2w template is used for 
                             aligning your data to this template)""")

    parser.add_argument('--affine_label_path',
                        default='fetal-affine-aligned-data/',
                        type=str,
                        help=r"""expects the input data to be in PATH + /sub-CC***_ses-***/ for each subject
                            This represents the path to the affinely registered Label data""")

    parser.add_argument('--output_path',
                        default='fetal-output-surfaces/',
                        type=str,
                        help=r"""location where it will store the predictions; 
                        note that it will create a folder per subject at this specified location""")

    parser.add_argument('--device',
                        default='cuda',
                        type=str,
                        help=r"""[cpu/cuda]
                        whether to run predictions on the cpu or GPU""")

    args = parser.parse_args()

    run_predict_all(args)


############################# BELOW are
############################# the MB models which were trained on bounti-43 for
############################# 1) 600E for WM on old surfaces + 50E on new surfaces with extra iterations
############################# 2) 800E for pial on old surfaces + 50E on new surfaces with extra iterations
############################# and saved in output-data-COTAN-wLAB-dhcp-MB-MBv4/
# python -m fetal-predict-all.fetal-dhcp-MB-predict-all-MBv2
# --tsv_file_subjects="/data/project/fetal-multibounti/fetal-dhcp-multibounti.tsv"
# --results_file="dhcp_MB_metrics_all.csv"
# --templates_path="/home/igr18/PycharmProjects/FetalSeg-CoTAN/templates/"
# --affine_label_path="/data/project/fetal-multibounti/3_Aff/"
# --output_path="/data/project/fetal-surfaces/surfaces-Irina/output-data-COTAN-wLAB-dhcp-MB-MBv4/"
# --device="cuda"