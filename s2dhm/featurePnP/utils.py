"""
A set of geometry tools for PyTorch tensors and sometimes NumPy arrays.
"""

import torch
import numpy as np
import torch.nn.functional as F


def to_homogeneous(points):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1]+(1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1]+(1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError


def from_homogeneous(points):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / points[..., -1:]


def batched_eye_like(x, n):
    """Create a batch of identity matrices.
    Args:
        x: a reference torch.Tensor whose batch dimension will be copied.
        n: the size of each identity matrix.
    Returns:
        A torch.Tensor of size (B, n, n), with same dtype and device as x.
    """
    return torch.eye(n).to(x)[None].repeat(len(x), 1, 1)


def create_norm_matrix(shift, scale):
    """Create a normalization matrix that shifts and scales points."""
    T = batched_eye_like(shift, 3)
    T[:, 0, 0] = T[:, 1, 1] = scale
    T[:, :2, 2] = shift
    return T


def normalize_keypoints(kpts, size=None, shape=None):
    """Normalize a set of 2D keypoints for input to a neural network.

    Perform the normalization according to the size of the corresponding
    image: shift by half and scales by the longest edge.
    Use either the image size or its tensor shape.

    Args:
        kpts: a batch of N D-dimensional keypoints: (B, N, D).
        size: a tensor of the size the image `[W, H]`.
        shape: a tuple of the image tensor shape `(B, C, H, W)`.
    """
    if size is None:
        assert shape is not None
        _, _, h, w = shape
        one = kpts.new_tensor(1)
        size = torch.stack([one*w, one*h])[None]

    shift = size.float() / 2
    scale = size.max(1).values.float() / 2  # actual SuperGlue mult by 0.7
    kpts = (kpts - shift[:, None]) / scale[:, None, None]

    T_norm = create_norm_matrix(shift, scale)
    T_norm_inv = create_norm_matrix(-shift/scale[:, None], 1./scale)
    return kpts, T_norm, T_norm_inv


def skew_symmetric(v):
    z = torch.zeros_like(v[..., 0])
    M = torch.stack([
        z, -v[..., 2], v[..., 1],
        v[..., 2], z, -v[..., 0],
        -v[..., 1], v[..., 0], z,
    ], dim=-1).reshape(v.shape[:-1]+(3, 3))
    return M


def T_to_E(T):
    """Convert batched poses (..., 4, 4) to batched essential matrices."""
    return T[..., :3, :3] @ skew_symmetric(T[..., :3, 3])


def sym_epipolar_distance(p0, p1, E):
    """Compute batched symmetric epipolar distances.
    Args:
        p0, p1: batched tensors of N 2D points of size (..., N, 2).
        E: essential matrices from camera 0 to camera 1, size (..., 3, 3).
    Returns:
        The symmetric epipolar distance of each point-pair: (..., N).
    """
    if p0.shape[-1] != 3:
        p0 = to_homogeneous(p0)
    if p1.shape[-1] != 3:
        p1 = to_homogeneous(p1)
    p1_E_p0 = torch.einsum('...ni,...ij,...nj->...n', p1, E, p0)
    E_p0 = torch.einsum('...ij,...nj->...ni', E, p0)
    Et_p1 = torch.einsum('...ij,...ni->...nj', E, p1)
    d = p1_E_p0**2 * (
        1. / (E_p0[..., 0]**2 + E_p0[..., 1]**2 + 1e-15) +
        1. / (Et_p1[..., 0]**2 + Et_p1[..., 1]**2 + 1e-15))
    return d


def so3exp_map(w):
    """Compute rotation matrices from batched twists.
    Args:
        w: batched 3D axis-angle vectors of size (..., 3).
    Returns:
        A batch of rotation matrices of size (..., 3, 3).
    """
    theta = w.norm(p=2, dim=-1, keepdim=True)
    W = skew_symmetric(w / theta)
    theta = theta[..., None]  # ... x 1 x 1
    res = W * torch.sin(theta) + (W @ W) * (1 - torch.cos(theta))
    res = torch.where(theta < 1e-12, torch.zeros_like(res), res)
    return torch.eye(3).to(W) + res


def bilinear_interpolation(grid, idx):
    # grid: C x H x W
    # idx: N x 2
    _, H, W = grid.shape
    x = idx[..., 0]
    y = idx[..., 1]
    x0 = torch.clamp(torch.floor(x), 0, W-1)
    y0 = torch.clamp(torch.floor(y), 0, H-1)
    x1 = torch.clamp(torch.ceil(x), 0, W-1)
    y1 = torch.clamp(torch.ceil(y), 0, H-1)
    weight00 = (x1 - x) * (y1 - y)
    weight01 = (x1 - x) * (y - y0)
    weight10 = (x - x0) * (y1 - y)
    weight11 = (x - x0) * (y - y0)
    x0 = x0.type(torch.LongTensor)
    y0 = y0.type(torch.LongTensor)
    x1 = x1.type(torch.LongTensor)
    y1 = y1.type(torch.LongTensor)
    grid00 = grid[..., y0, x0]
    grid01 = grid[..., y0, x1]
    grid10 = grid[..., y1, x0]
    grid11 = grid[..., y1, x1]
    # print(weight00)
    # print(weight01)
    # print(weight10)
    # print(weight11)

    return weight00*grid00 + weight01*grid01 + weight10*grid10 + weight11*grid11

def sobel_filter(f):
    # f: BxCxHxW

    b, c, h, w = f.shape
    sobel_y = torch.FloatTensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3).to(f)
    sobel_x = torch.FloatTensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).to(f)
    f_gradx = F.conv2d(f.view(-1, 1, h, w), sobel_x, stride=1, padding=1).view(b, c, h, w)
    f_grady = F.conv2d(f.view(-1, 1, h, w), sobel_y, stride=1, padding=1).view(b, c, h, w)
    return f_gradx, f_grady

def np_gradient_filter(f):
    # f: BxCxHxW

    b, c, h, w = f.shape
    np_gradient_y = torch.FloatTensor([[0., -0.5, 0.], [0., 0., 0.], [0., 0.5, 0.]]).view(1, 1, 3, 3).to(f)
    np_gradient_x = torch.FloatTensor([[0., 0., 0], [-0.5, 0., 0.5], [0., 0., 0]]).view(1, 1, 3, 3).to(f)
    f_gradx = F.conv2d(f.view(-1, 1, h, w), np_gradient_x, stride=1, padding=1).view(b, c, h, w)
    f_grady = F.conv2d(f.view(-1, 1, h, w), np_gradient_y, stride=1, padding=1).view(b, c, h, w)
    return f_gradx, f_grady



# if __name__ == "__main__":
#     a = torch.ones((1, 2, 2))
#     a[:, 0, 1] = 2
#     a[:, 1, 0] = 3
#     a[:, 1, 1] = 4
# 
#     idx = torch.from_numpy(np.random.rand(4,2))
#     print(idx)
# 
#     print(bilinear_interpolation(a, idx))