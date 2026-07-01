########################################################################
###### IRINA GRIGORESCU
######
###### This contains code to calculate metrics on meshes
########################################################################

import numpy as np
from scipy.spatial import cKDTree
import torch

from src.inflate import mris_inflate
from src.mesh import (
    laplacian_smooth,
    vert_normal,
    adjacency_matrix
)


def calculate_average_thickness(vert_inner_surface, vert_outer_surface, return_surface=False):
    """
    Calculate average thickness between inner surface (WM) to the outer surface (pial)

    :param vert_inner_surface: vertices of the inner surface such as white matter surface (|V|,)
    :param vert_outer_surface: vertices of the outer surface such as pial surface (|V|,)
    :param return_surface: True/False if to return cortical thickness, (|V|) numpy.array

    :return: d_mean, d_std, d_median, d_min, d_max, d_in2out, d_out2in, [thickness (|V|,)] (optional)
    """
    tree_outer = cKDTree(vert_outer_surface, leafsize=20)
    tree_inner = cKDTree(vert_inner_surface, leafsize=20)

    dist_inner2outer, _ = tree_outer.query(vert_inner_surface, k=1)
    dist_outer2inner, _ = tree_inner.query(vert_outer_surface, k=1)

    dist_all = np.concatenate([dist_inner2outer, dist_outer2inner])

    d_mean = float(np.mean(dist_all))
    d_std  = float(np.std(dist_all))
    d_median = float(np.median(dist_all))
    d_min = float(np.min(dist_all))
    d_max = float(np.max(dist_all))
    d_in2out = float(np.mean(dist_inner2outer))
    d_out2in = float(np.mean(dist_outer2inner))

    if return_surface:
        thickness = 0
        thickness += dist_outer2inner / 2.0
        thickness += dist_inner2outer / 2.0

        return d_mean, d_std, d_median, d_min, d_max, d_in2out, d_out2in, thickness

    else:
        return d_mean, d_std, d_median, d_min, d_max, d_in2out, d_out2in



###################################################
################################################### Please check individual repositories for original code below
###################################################
def metric_dilation(metric, face, roi=None, n_iters=10):
    """
    Metric dilation within the region of interest.
    For original code please see: https://github.com/m-qiang/dhcp-dl-neonatal

    :param metric: surface metric, (1,|V|,1) torch.Tensor
    :param face: mesh faces, (1,|V|,3) torch.LongTensor
    :param roi: region of interest, (|V|) numpy.array
    :param n_iters: number of dilation iterations, int

    :return: metric: dilated surface metric, (1,|V|,1) torch.Tensor
    """

    # compute adjacency matrix
    A = adjacency_matrix(face)
    for n in range(n_iters):
        # find all nonzero metric values
        metric_nonzero = 1. - (metric == 0).float()
        # weighted without the zero metric
        degree = A.bmm(metric_nonzero) + 1e-8
        # only update the metrics with zero values
        metric += (1. - metric_nonzero) * A.bmm(metric) / degree
    metric = metric[0, :, 0].cpu().numpy()
    if roi is not None:
        metric = metric * roi
    return metric


def curvature(vert, face, curv_type='mean', smooth_iters=0):
    """
    Estimate curvature of the surface.
    For original code please see: https://github.com/m-qiang/dhcp-dl-neonatal

    This function reimplements the method in the connectome workbench commandline.
    For original code please see:
    https://github.com/Washington-University/workbench/blob/master/src/Algorithms/AlgorithmSurfaceCurvature.cxx

    :param vert: mesh vertices, (1,|V|,3) torch.Tensor
    :param face: mesh faces, (1,|F|,3) torch.LongTensor
    :param curv_type: ['mean', 'gaussian']
    :param smooth_iters: number of smoothing iterations

    :return: curv: curvature, (|V|) numpy.array
    """

    n_vert = vert.shape[1]
    normal = vert_normal(vert, face)
    basis = (normal[:, :, 0].abs() > normal[:, :, 1].abs()).unsqueeze(-1).float()
    basis = torch.cat([1 - basis, basis, torch.zeros_like(basis)], dim=-1)
    ihat = torch.cross(normal, basis, dim=-1)
    ihat = ihat / ihat.norm(dim=-1, keepdim=True)
    jhat = torch.cross(normal, ihat, dim=-1)
    edge = torch.cat([face[0, :, [0, 1]],
                      face[0, :, [1, 2]],
                      face[0, :, [2, 0]]], dim=0).T  # compute edges

    # edge[0]: center vertex, edge[1]: neighborhood vertex
    neigh_normal = normal[:, edge[1]]  # find normals for neighborhoods
    neigh_diff = vert[:, edge[1]] - vert[:, edge[0]]
    ihat = ihat[:, edge[0]]
    jhat = jhat[:, edge[0]]

    norm_proj_0 = (neigh_normal * ihat).sum(-1)
    norm_proj_1 = (neigh_normal * jhat).sum(-1)
    diff_proj_0 = (neigh_diff * ihat).sum(-1)
    diff_proj_1 = (neigh_diff * jhat).sum(-1)

    sig_x = diff_proj_0 * diff_proj_0
    sig_xy = diff_proj_0 * diff_proj_1
    sig_y = diff_proj_1 * diff_proj_1
    norm_x = norm_proj_0 * diff_proj_0
    norm_xy = norm_proj_0 * diff_proj_1 + norm_proj_1 * diff_proj_0
    norm_y = norm_proj_1 * diff_proj_1

    # build adjacency matrix
    values = torch.cat([sig_x, sig_xy, sig_y, norm_x, norm_xy, norm_y]).T
    neigh_matrix = torch.sparse_coo_tensor(
        edge, values, (n_vert, n_vert, 6)).unsqueeze(0)
    # sum all neighbors
    values_per_vertex = torch.sparse.sum(neigh_matrix, dim=-2).to_dense()
    sig_x = values_per_vertex[..., 0]
    sig_xy = values_per_vertex[..., 1]
    sig_y = values_per_vertex[..., 2]
    norm_x = values_per_vertex[..., 3]
    norm_xy = values_per_vertex[..., 4]
    norm_y = values_per_vertex[..., 5]

    sig_xy2 = sig_xy * sig_xy
    denom = (sig_x + sig_y) * (-sig_xy2 + sig_x * sig_y)
    denom_ = denom + 1e-8  # avoid divide by 0

    a = (norm_x * (-sig_xy2 + sig_x * sig_y + sig_y * sig_y) -
         norm_xy * sig_xy * sig_y + norm_y * sig_xy2) / denom_
    b = (-norm_x * sig_xy * sig_y + norm_xy * sig_x * sig_y -
         norm_y * sig_x * sig_xy) / denom_
    c = (norm_x * sig_xy2 - norm_xy * sig_x * sig_xy +
         norm_y * (sig_x * sig_x - sig_xy2 + sig_x * sig_y)) / denom_
    trC = a + c
    detC = a * c - b * b
    temp = trC * trC - 4 * detC
    delta = temp.abs().sqrt()
    k1 = (trC + delta) / 2
    k2 = (trC - delta) / 2

    # set curvature to zero if denom=0 or temp<0
    k1[torch.where(denom == 0)] = 0.
    k2[torch.where(denom == 0)] = 0.
    k1[torch.where(temp < 0)] = 0.
    k2[torch.where(temp < 0)] = 0.

    if curv_type == 'gaussian':
        curv = k1 * k2
    elif curv_type == 'mean':
        curv = (k1 + k2) / 2

    # smooth the curvature
    curv = laplacian_smooth(
        curv.unsqueeze(-1), face, lambd=1.0, n_iters=smooth_iters)
    return curv[0, :, 0].cpu().numpy()


def sulcal_depth(vert, face, nsteps=10, return_vertices=False, verbose=False):
    """
    Estimate sulcal depth by inflating the surface.
    For original code please see: https://github.com/m-qiang/dhcp-dl-neonatal

    :param vert: mesh vertices, (1,|V|,3) torch.Tensor
    :param face: mesh faces, (1,|F|,3) torch.LongTensor
    :param nsteps: number of steps to estimate sulcal depth
    :param return_vertices: whether to return vertices as well as sulcal depth map
    :param verbose: if report

    :return: inflatedvert: inflated vertices, (1,|V|,3) torch.Tensor
             sulc: sulcal depth, (|V|) numpy.array
    """

    inflatedvert, sulc = mris_inflate(
        vert, face, n_steps=nsteps, track_sulcal_depth=True, verbose=verbose)

    ## remember that inflated vert are pytorch,
    ## so you need to do inflatedvert[0].cpu().numpy()
    if return_vertices:
        return inflatedvert, sulc
    else:
        return sulc