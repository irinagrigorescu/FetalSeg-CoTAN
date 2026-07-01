########################################################################
###### IRINA GRIGORESCU
######
###### These are mesh / vertices specific functions.
######
###### Some code from https://github.com/m-qiang/dhcp-dl-neonatal/
########################################################################


import torch
import numpy as np
from scipy.sparse import coo_matrix


def vert_normal(v, f):
    """
    Compute the normal vector of each vertex.

    For original code please see: https://github.com/m-qiang/dhcp-dl-neonatal/blob/main/utils/mesh.py

    This function is retrieved from pytorch3d.
    For original code please see:
    _compute_vertex_normals function in
    https://pytorch3d.readthedocs.io/en/latest/
    _modules/pytorch3d/structures/meshes.html

    :param v: input mesh vertices, (1,|V|,3) torch.Tensor
    :param f: input mesh    faces, (1,|F|,3) torch.LongTensor

    :return: v_normal: vertex normals, (1,|V|,3) torch.Tensor
    """

    n_v = torch.zeros_like(v)  # normals of vertices
    v_f = v[:, f[0]]

    # compute normals of faces
    n_f_0 = torch.cross(v_f[:, :, 1] - v_f[:, :, 0], v_f[:, :, 2] - v_f[:, :, 0], dim=2)
    n_f_1 = torch.cross(v_f[:, :, 2] - v_f[:, :, 1], v_f[:, :, 0] - v_f[:, :, 1], dim=2)
    n_f_2 = torch.cross(v_f[:, :, 0] - v_f[:, :, 2], v_f[:, :, 1] - v_f[:, :, 2], dim=2)

    # sum the faces normals
    n_v = n_v.index_add(1, f[0, :, 0], n_f_0)
    n_v = n_v.index_add(1, f[0, :, 1], n_f_1)
    n_v = n_v.index_add(1, f[0, :, 2], n_f_2)

    n_v = n_v / (torch.norm(n_v, dim=-1).unsqueeze(-1) + 1e-12)

    return n_v


def mesh_area(vert, face):
    """
    Compute the total area of the mesh
    For original code please see: https://github.com/m-qiang/dhcp-dl-neonatal/blob/main/utils/mesh.py

    :param vert: input mesh vertices, (1,|V|,3) torch.Tensor
    :param face: input mesh faces, (1,|F|,3) torch.LongTensor

    :return: area: mesh area, float
    """

    v0 = vert[:, face[0, :, 0]]
    v1 = vert[:, face[0, :, 1]]
    v2 = vert[:, face[0, :, 2]]
    area = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)
    return area.sum().item()


def face_area(vert, face):
    """
    Compute the area of each face
    For original code please see: https://github.com/m-qiang/dhcp-dl-neonatal/blob/main/utils/mesh.py

    :param vert: input mesh vertices, (1,|V|,3) torch.Tensor
    :param face: input mesh faces, (1,|F|,3) torch.LongTensor

    :return: area: face area, (|F|,3) torch.Tensor
    """

    v0 = vert[:, face[0, :, 0]]
    v1 = vert[:, face[0, :, 1]]
    v2 = vert[:, face[0, :, 2]]
    area = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)
    return area[0]


def adjacency_matrix(face):
    """
    Compute adjacency matrix.
    For original code please see: https://github.com/m-qiang/dhcp-dl-neonatal/blob/main/utils/mesh.py

    :param face: input mesh faces, (1,|F|,3) torch.LongTensor

    :return: A: adjacency matrix, (1,|V|,|V|) torch.sparse.Tensor
    """

    nv = face.max().item() + 1
    edge = torch.cat([face[0, :, [0, 1]],
                      face[0, :, [1, 2]],
                      face[0, :, [2, 0]]], dim=0).T
    # adjacency matrix A
    A = torch.sparse_coo_tensor(
        edge, torch.ones_like(edge[0]).float(), (nv, nv)).unsqueeze(0)
    # number of neighbors for each vertex
    # adj_degree = torch.sparse.sum(A, dim=-1).to_dense().unsqueeze(-1)
    return A  # , adj_degree


def laplacian(face):
    """
    Compute Laplacian matrix.
    For original code please see: https://github.com/m-qiang/dhcp-dl-neonatal/blob/main/utils/mesh.py

    :param face: input mesh faces, (1,|F|,3) torch.LongTensor

    :return: L: Laplacian matrix, (1,|V|,|V|) torch.sparse.Tensor
    """
    nv = face.max().item() + 1
    edge = torch.cat([face[0, :, [0, 1]],
                      face[0, :, [1, 2]],
                      face[0, :, [2, 0]]], dim=0).T
    # adjacency matrix A
    A = torch.sparse_coo_tensor(
        edge, torch.ones_like(edge[0]).float(), (nv, nv)).unsqueeze(0)

    # number of neighbors for each vertex
    degree = torch.sparse.sum(A, dim=-1).to_dense()[0]
    weight = 1. / degree[edge[0]]
    # normalized adjacency matrix
    A_hat = torch.sparse_coo_tensor(
        edge, weight, (nv, nv)).unsqueeze(0)

    # normalized degree matrix, i.e., identity matrix
    # set the diagonal entries to one
    self_edge = torch.arange(nv)[None].repeat([2, 1]).to(face.device)
    D_hat = torch.sparse_coo_tensor(
        self_edge, torch.ones_like(self_edge[0]).float(), (nv, nv)).unsqueeze(0)
    L = D_hat - A_hat
    return L


def laplacian_smooth(vert, face, lambd=1., n_iters=1):
    """
    Laplacian mesh smoothing.
    For original code please see: https://github.com/m-qiang/dhcp-dl-neonatal/blob/main/utils/mesh.py

    :param vert: input mesh vertices, (1,|V|,3) torch.Tensor
    :param face: input mesh faces, (1,|F|,3) torch.LongTensor
    :param lambd: strength of mesh smoothing [0,1]
    :param n_iters: number of mesh smoothing iterations

    :return: vert: smoothed mesh vertices, (1,|V|,3) torch.Tensor
    """
    L = laplacian(face)
    for n in range(n_iters):
        vert = vert - lambd * L.bmm(vert)
    return vert


def neighbor_matrix(face, n_neighbors=2):
    """
    Compute n-neighborhood (n-hop) adjacency matrix.
    For original code please see: https://github.com/m-qiang/dhcp-dl-neonatal/blob/main/utils/mesh.py

    :param face: input mesh faces, (1,|F|,3) torch.LongTensor
    :param n_neighbors: number of hops, int

    :return: A_n: n-hop adjacency matrix, (1,|V|,|V|) torch.sparse.Tensor
    """

    f = face[0].cpu().numpy()
    nv = face.max().item() + 1

    # create initial adjacency matrix
    edge = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    A = coo_matrix((np.ones(edge.shape[0], dtype=np.int8),
                    (edge[:, 0], edge[:, 1])), shape=(nv, nv))

    # compute connection matrix
    connect_matrix = A
    for n in range(n_neighbors - 1):
        connect_matrix = connect_matrix.dot(connect_matrix)
        connect_matrix = (connect_matrix > 0).astype(np.int8).tocoo()
        # remove diagonal elements (self-connection)
        connect_matrix.setdiag(np.zeros(nv))
        connect_matrix.eliminate_zeros()

    # create connection matrix An
    edge_n = np.stack([connect_matrix.row, connect_matrix.col])
    edge_n = torch.LongTensor(edge_n).to(face.device)
    A_n = torch.sparse_coo_tensor(
        edge_n, torch.ones_like(edge_n[0]).float(), (nv, nv)).unsqueeze(0)
    return A_n


def taubin_smooth(vert, face, lambd=0.5, mu=-0.53, n_iters=1):
    """
    Taubin mesh smoothing.
    For original code please see: https://github.com/m-qiang/CoSeg/blob/main/utils/mesh.py

    :param vert: input mesh vertices, (1,|V|,3) torch.Tensor
    :param face: input mesh faces, (1,|F|,3) torch.LongTensor
    :param lambd: strength of mesh smoothing [0,1]
    :param mu: strength of mesh smoothing [-1,0]
    :param n_iters: number of mesh smoothing iterations

    :return: vert: smoothed mesh vertices, (1,|V|,3) torch.Tensor
    """
    L = laplacian(face)
    for n in range(n_iters):
        vert = vert - mu * L.bmm(vert)
        vert = vert - lambd * L.bmm(vert)
    return vert


def apply_affine(vert, affine):
    """
    Apply affine transformation to surface vertices.

    For original code please see: https://github.com/m-qiang/dhcp-dl-neonatal/blob/main/utils/mesh.py

    Inputs:
    :param vert: mesh vertices, (|V|,3) numpy.array
    :param affine: affine matrix, (4,4) numpy.array

    :return: vertices after affine transform, (|V|,3) numpy.array
    """
    return vert @ affine[:3, :3].T + affine[:3, -1]

