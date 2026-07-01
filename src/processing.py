########################################################################
###### IRINA GRIGORESCU
######
###### This file contains pre- and post- processing utilities for data/surfaces
########################################################################

import torch
import numpy as np
from src.mesh import apply_affine


def preprocess_surfaces(surf_in, hemi, affine_in, flag="init"):
    """
    Preprocessing needed

    :param surf_in:
    :param hemi:
    :param affine_in:
    :param flag:
    :return:
    """
    # ------- READ IN VERTICES AND FACES
    v_in, f_in = surf_in.agg_data('pointset'), surf_in.agg_data('triangle')
    # ------- APPLY AFFINE TO VERTICES if flag == init
    if flag == "init":
        v_in = apply_affine(v_in, np.linalg.inv(affine_in))
    # ------- RE-CENTRE       VERTICES
    if hemi == "left" and flag == "init":
        v_in[:, 0] = v_in[:, 0] - 70
        v_in[:, 1] = v_in[:, 1] - 4
        v_in[:, 2] = v_in[:, 2] - 4
    # ------- NORMALISE       VERTICES
    v_in = (v_in - [56, 112, 80]) / 112
    # ------- RE-ORDER FACES BECAUSE COUNTERCLOCKWISE ¯\_(ツ)_/¯
    f_in = f_in[:, [2, 1, 0]]
    # ------- MAKE THEM INTO TENSORS
    v_in = torch.Tensor(v_in[None])
    f_in = torch.LongTensor(f_in[None])

    return v_in, f_in


def postprocess_surfaces(vertices_wm_in, vertices_p_in, faces_surface, hemi, affine_orig):
    # ------- MAKE TENSORS INTO NUMPY
    vertices_wm_ = vertices_wm_in[0].cpu().numpy().copy()
    vertices_p_ = vertices_p_in[0].cpu().numpy().copy()
    faces_in_ = faces_surface[0].cpu().numpy().copy()
    # ------- RE-ORDER FACES BACK ¯\_(ツ)_/¯
    faces_in_ = faces_in_[:, [2, 1, 0]]

    # ------- MAP SURFACES TO THEIR ORIGINAL SPACES
    if hemi == "right":
        vertices_wm_ = vertices_wm_ * 112 + [56, 112, 80]
        vertices_p_ = vertices_p_ * 112 + [56, 112, 80]
    else:
        vertices_wm_ = vertices_wm_ * 112 + [56, 112, 80]
        vertices_wm_[:, 0] = vertices_wm_[:, 0] + 64
        vertices_p_ = vertices_p_ * 112 + [56, 112, 80]
        vertices_p_[:, 0] = vertices_p_[:, 0] + 64

    vertices_wm_ = apply_affine(vertices_wm_, affine_orig)
    vertices_p_ = apply_affine(vertices_p_, affine_orig)

    # ------- RETURN EVERYTHING
    return vertices_wm_, vertices_p_, faces_in_