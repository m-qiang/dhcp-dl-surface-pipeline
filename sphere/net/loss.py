import torch
import numpy as np
from utils.mesh import face_area


def distortion(metric_sphere, metric_surf):
    """
    Compute metric distortion.

    Inputs:
    - metric_pred: the metric of the (prediced) sphere, (1,|V|) torch.Tensor
    - metric_gt: the metric of the reference WM surface, (1,|V|) torch.Tensor
    
    Returns:
    - distort: the metric distortion ratio, torch.float
    """
    ratio = metric_sphere / (metric_surf + 1e-12)
    # scaling coeffcient
    beta = ratio.mean() / (ratio ** 2).mean()
    distort = ((beta * ratio - 1)**2).mean().sqrt()
    return distort


def edge_distortion(vert_sphere, vert_surf, edge):
    """
    Compute edge distortion.

    Inputs:
    - vert_sphere: the vertices of the (prediced) sphere, (1,|V|) torch.Tensor
    - vert_surf: the vertices of the reference WM surface, (1,|V|) torch.Tensor
    - edge: the edge list of the mesh, (2,|E|) torch.LongTensor
    
    Returns:
    - the metric distortion ratio, torch.float
    """
    # compute edge length
    edge_len_sphere = (vert_sphere[:,edge[0]] -\
                       vert_sphere[:,edge[1]]).norm(dim=-1)
    edge_len_surf = (vert_surf[:,edge[0]] -\
                     vert_surf[:,edge[1]]).norm(dim=-1)
    return distortion(edge_len_sphere, edge_len_surf)


def area_distortion(vert_sphere, vert_surf, face):
    """
    Compute area distortion.

    Inputs:
    - vert_sphere: the vertices of the (prediced) sphere, (1,|V|) torch.Tensor
    - vert_surf: the vertices of the reference WM surface, (1,|V|) torch.Tensor
    - face: the mesh faces, (1,|F|,3) torch.LongTensor
    
    Returns:
    - the metric distortion ratio, torch.float
    """
    
    # compute face area
    area_sphere = face_area(vert_sphere, face)
    area_surf = face_area(vert_surf, face)
    return distortion(area_sphere, area_surf)


def geodesic_distortion(vert_pred, vert_gt, n_samples=500000, radius=1.):
    """
    Compute geodesic distortion between two spheres.

    Inputs:
    - vert_pred: the vertices of the prediced sphere, (1,|V|) torch.Tensor
    - vert_gt: the vertices of the ground truth sphere, (1,|V|) torch.Tensor
    - n_samples: the number of randomly sampled pairs of vertices, int
    - radius: the radius of the sphere, float
    
    Returns:
    - the metric distortion ratio, torch.float
    """
    
    n_vert = vert_pred.shape[1]
    # randomly sample vertices pairs
    idx_i = np.random.choice(n_vert, n_samples, replace=True)
    idx_j = np.random.choice(n_vert, n_samples, replace=True)
    vert_pred_ = vert_pred / radius
    vert_gt_ = vert_gt / radius
    vert_pred_i = vert_pred_[:,idx_i]
    vert_pred_j = vert_pred_[:,idx_j]
    vert_gt_i = vert_gt_[:,idx_i]
    vert_gt_j = vert_gt_[:,idx_j]

    # compute arc length as geodesic distance
    d_pred = ((vert_pred_i - vert_pred_j)**2).sum(-1)
    d_gt = ((vert_gt_i - vert_gt_j)**2).sum(-1)
    d_pred = radius * torch.acos(torch.clamp(
        1.0-d_pred/2, min=-1+1e-6, max=1-1e-6))+1e-8
    d_gt = radius * torch.acos(torch.clamp(
        1.0-d_gt/2, min=-1+1e-6, max=1-1e-6))+1e-8
    
    ratio = d_pred / (d_gt + 1e-12)
    distort = ((ratio - 1)**2).mean().sqrt()
    return distort
