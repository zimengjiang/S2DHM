import torch
from torch import nn

from featurePnP.utils import (to_homogeneous, from_homogeneous, batched_eye_like,
                    skew_symmetric, so3exp_map, sobel_filter, np_gradient_filter, bilinear_interpolation)
from featurePnP.losses import scaled_loss, squared_loss, pose_error

from PIL import Image
import scipy.io
import numpy as np
import logging

from pose_prediction.matrix_utils import matrix_quaternion

def optimizer_step(g, H, lambda_=0):
    """One optimization step with Gauss-Newton or Levenberg-Marquardt.
    Args:
        g: batched gradient tensor of size (..., N).
        H: batched hessian tensor of size (..., N, N).
        lambda_: damping factor for LM (use GN if lambda_=0).
    """
    if lambda_:  # LM instead of GN
        D = (H.diagonal(dim1=-2, dim2=-1) + 1e-9).diag_embed()
        H = H + D*lambda_
    try:
        P = torch.inverse(H)
    except RuntimeError as e:
        logging.warning(f'Determinant: {torch.det(H)}')
        raise e
    delta = -(P @ g[..., None])[..., 0]
    return delta

class FeaturePnP(nn.Module):
    def __init__(self, iterations, device, loss_fn=squared_loss, init_lambda=0.01, verbose=False, min_lambda=1e-6, max_lambda=1e2):
        super().__init__()
        self.iterations = iterations
        self.device = device
        self.loss_fn = loss_fn
        self.verbose = verbose
        self.lambda_ = init_lambda
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda


    def forward(self, 
                query_prediction, 
                reference_prediction, 
                local_reconstruction, 
                mask,
                query_dense_hypercolumn, 
                reference_dense_hypercolumn,
                query_intrinsics, 
                size_ratio,
                points_2D,
                R_gt=None, 
                t_gt=None,
                use_sobel=True):
        # TODO: take distCoeffs into account using cv2.undistortPoints
        with torch.no_grad():
            q_matrix = torch.FloatTensor(query_prediction.matrix).to(self.device)
            r_matrix = torch.FloatTensor(reference_prediction.matrix).to(self.device)
            r_matrix_inv = r_matrix.inverse()

            # from pts0 to pts1: q_matrix @ r_matrix_inv
            relative_matrix = q_matrix @ r_matrix_inv

            R_init = relative_matrix[:3, :3]
            t_init = relative_matrix[:3, 3]
            imgf0 = reference_dense_hypercolumn
            imgf1 = query_dense_hypercolumn
            # determine which filter for calculating numerical image gradient
            # empirically the 2nd order center difference (np_gradient_filter) works better 
            if use_sobel:
                imgf1_gx, imgf1_gy = sobel_filter(imgf1) 
            else:
                imgf1_gx, imgf1_gy = np_gradient_filter(imgf1) 
            # remove batch dimension since it is always 1
            imgf1_gx = imgf1_gx.squeeze()
            imgf1_gy = imgf1_gy.squeeze()
            imgf0 = imgf0.squeeze()
            imgf1 = imgf1.squeeze()

            # 0 refers to the reference image and 1 refers to the query image
            K0 = torch.FloatTensor(local_reconstruction.intrinsics).to(self.device)
            K1 = torch.FloatTensor(query_intrinsics).to(self.device)

            lambda_ = self.lambda_

            # already in pixel coordinates
            pts_2d_0 = torch.FloatTensor(local_reconstruction.points_2D[mask]).to(self.device) # N x 2  
            # img0_idx = torch.floor(pts_2d_0 / size_ratio).type(torch.LongTensor)
            # use bilinear interpolation instead of floor
            img0_idx = (pts_2d_0 / size_ratio)
            # extracted_feat0 = (imgf0[:, img0_idx[:, 1], img0_idx[:, 0]]).transpose(0, 1)
            extracted_feat0 = bilinear_interpolation(imgf0, img0_idx).transpose(0, 1)

            # pts_3d_0 = to_homogeneous(local_reconstruction.points_3D[mask]) @ r_matrix_inv.T
            pts_3d_0 = from_homogeneous(
                to_homogeneous(torch.FloatTensor(local_reconstruction.points_3D[mask])).to(self.device) 
                @ r_matrix.T)

            R = R_init
            t = t_init

            # for us this should be one for all points since we have no idea which keypoint is more important
            scale = torch.ones((pts_2d_0.shape[0],)).type(torch.FloatTensor).to(self.device)
            for i in range(self.iterations):
                pts_3d_1 = pts_3d_0 @ R.T + t
                pts_2d_1 = from_homogeneous(pts_3d_1 @ K1.T)
                # img1_idx = torch.floor(pts_2d_1 / size_ratio).type(torch.LongTensor).to(self.device)
                img1_idx = pts_2d_1 / size_ratio
                # img1_idx might be slightly off the boundary
                # For good initialization, we don't have this issue of out of boundary keypoints
                img1_idx[:, 0].clamp_(0, imgf1.shape[2]-1)
                img1_idx[:, 1].clamp_(0, imgf1.shape[1]-1)
                # extracted_feat1 = (imgf1[:, img1_idx[:, 1], img1_idx[:, 0]]).transpose(0, 1)
                extracted_feat1 = bilinear_interpolation(imgf1, img1_idx).transpose(0, 1)

                error = extracted_feat1 - extracted_feat0
                cost = (error**2).sum(-1)
                cost, weights, _ = scaled_loss(cost, self.loss_fn, scale)
                if i == 0:
                    prev_cost = cost.mean(-1)
                if self.verbose and i % 9 == 0:
                    print('Iter {} loss {}'.format(i, cost.mean().item()))

                # calculate gradient for LM
                J_p_T = torch.cat([
                    batched_eye_like(pts_3d_1, 3), -skew_symmetric(pts_3d_1,)
                ], -1)

                shape = pts_3d_1.shape[:-1]
                o, z = pts_3d_1.new_ones(shape), pts_3d_1.new_zeros(shape)
                fx = K1[0, 0]
                fy = K1[1, 1]
                J_h_p = torch.stack([
                    o*fx,   z,      -fx*pts_3d_1[..., 0] / pts_3d_1[..., 2],
                    z,      o*fy,   -fy*pts_3d_1[..., 1] / pts_3d_1[..., 2],
                ], dim=-1).reshape(shape+(2, 3)) / pts_3d_1[..., 2, None, None] / size_ratio

                # J_e_hx = (imgf1_gx[:, img1_idx[:, 1], img1_idx[:, 0]]).transpose(0, 1)
                # J_e_hy = (imgf1_gy[:, img1_idx[:, 1], img1_idx[:, 0]]).transpose(0, 1)
                J_e_hx = bilinear_interpolation(img1gx, img1_idx).transpose(0, 1)
                J_e_hy = bilinear_interpolation(img1gy, img1_idx).transpose(0, 1)
                J_e_h = torch.stack([J_e_hx, J_e_hy], dim=-1)
                J_e_T = J_e_h @ J_h_p @ J_p_T

                Grad = torch.einsum('...ijk,...ij->...ik', J_e_T, error)
                Grad = weights[..., None] * Grad
                Grad = Grad.sum(-2)  # Grad was ... x N x 6
                J = J_e_T
                Hess = torch.einsum('...ijk,...ijl->...ikl', J, J)
                Hess = weights[..., None, None] * Hess
                Hess = Hess.sum(-3)  # Hess was ... x N x 6 x 6

                delta = optimizer_step(Grad, Hess, lambda_)
                if torch.isnan(delta).any():
                    logging.warning('NaN detected, exit')
                    break
                dt, dw = delta[..., :3], delta[..., 3:6]
                dr = so3exp_map(dw)
                R_new = dr @ R
                t_new = dr @ t + dt

                new_pts_3d_1 = pts_3d_0 @ R_new.T + t_new
                new_pts_2d_1 = from_homogeneous(new_pts_3d_1 @ K1.T)
                # new_img1_idx = torch.floor(new_pts_2d_1 / size_ratio).type(torch.LongTensor).to(self.device)
                new_img1_idx = new_pts_2d1 / size_ratio
                new_img1_idx[:, 0].clamp_(0, imgf1.shape[2]-1)
                new_img1_idx[:, 1].clamp_(0, imgf1.shape[1]-1)
                # new_extracted_feat1 = (imgf1[:, new_img1_idx[:, 1], new_img1_idx[:, 0]]).transpose(0, 1)
                new_extracted_feat1 = bilinear_interpolation(imgf1, new_img1_idx).transpose(0, 1)

                new_error = new_extracted_feat1 - extracted_feat0 
                new_cost = (new_error**2).sum(-1)
                new_cost = scaled_loss(new_cost, self.loss_fn, scale)[0].mean()

                if new_cost > prev_cost and lambda_ == self.max_lambda:
                    print("Stop at iteration {}".format(i)) 
                    break
                lambda_ = np.clip(lambda_ * (10 if new_cost > prev_cost else 1/10),
                                  self.min_lambda, self.max_lambda)

                if new_cost > prev_cost:  # cost increased
                    continue
                
                best_pts_2d_1 = new_pts_2d_1
                prev_cost = new_cost
                R, t = R_new, t_new

                if R_gt is not None and t_gt is not None:
                    if self.verbose:
                        abs_pose = self.relative_to_abs_pose(R, t, r_matrix)
                        r_error, t_error = pose_error(abs_pose[:3, :3], abs_pose[:3, 3], R_gt, t_gt)
                        print('Pose error:', r_error.cpu().numpy(), t_error.cpu().numpy())
            
            # update query prediciton's quaterion and matrix
            self.update_prediction(
                pts_3D = local_reconstruction.points_3D[mask],
                # pts_2D = torch.round(pts_2d_1),
                # pts_2D = points_2D, 
                # bug fix: here should use updated 2d keypoints in the query image
                # pts_2D = torch.round(best_pts_2d_1).type(torch.LongTensor).cpu().numpy(),
                pts_2D = torch.round(best_pts_2d_1),
                pose = self.relative_to_abs_pose(R, t, r_matrix),
                prediction = query_prediction,
                reference_pts_2D = local_reconstruction.points_2D[mask],
                query_intrinsics = K1,
                reprojectionThres = 12.0,
            )

            # return R, t

    def relative_to_abs_pose(self, R, t, r_matrix):
        abs_pose = torch.zeros((4, 4), dtype=torch.float32, device=self.device)
        abs_pose[:3, :3] = R
        abs_pose[:3, 3] = t
        abs_pose[3, 3] = 1
        abs_pose = abs_pose @ r_matrix
        return abs_pose


    def update_prediction(self, pts_3D, pts_2D, pose, prediction, reference_pts_2D, query_intrinsics, reprojectionThres=12.0):
        # points_3D_proj = np.round(from_homogeneous(from_homogeneous(to_homogeneous(points_3D)  @ prediction.matrix.T) @ local_reconstruction.intrinsics.T))
        # pts_3D_proj = from_homogeneous(to_homogeneous(torch.from_numpy(pts_3D.astype(np.float32))).cuda() @ pose.T)
        pts_3D_cuda = torch.from_numpy(pts_3D.astype(np.float32)).cuda()
        pts_3D_proj = from_homogeneous(from_homogeneous(to_homogeneous(pts_3D_cuda) @ pose.T) @ query_intrinsics.T)
        dist = torch.sum(torch.pow(pts_3D_proj - pts_2D, 2), axis=-1)
        # print(pts_3D_proj.cpu().numpy(), pts_2D)
        inliers = (dist < reprojectionThres * reprojectionThres).cpu().numpy()
        prediction.num_inliers = np.sum(inliers)
        prediction.reference_inliers = reference_pts_2D[inliers]
        prediction.query_inliers = pts_2D[inliers]
        prediction.quaternion = matrix_quaternion(pose.cpu().numpy()) 
        prediction.matrix = pose.cpu().numpy()
        return


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def keypoints2example(img_idx0):
    poses = scipy.io.loadmat('data/checkerboard/poses.mat')['poses']
    poses = np.concatenate((poses, np.zeros((210, 1, 4))), axis=1)
    poses[:, 3, 3] = 1
    depths = scipy.io.loadmat('data/checkerboard/depths.mat')['depths']
    pts2d = scipy.io.loadmat('data/checkerboard/pts2d.mat')['pts2d']
    pts3d = scipy.io.loadmat('data/checkerboard/pts3d.mat')['p_W_corners']

    imgf0 = skimage.io.imread("data/checkerboard/images_undistorted/img_{:04d}.jpg".format(img_idx0+1)).astype(np.float32)
    K = np.array([[420.506712, 0.        ,  355.208298],
                  [0.        , 420.610940,  250.336787],
                  [0.        , 0.        ,  1.]])
    pts2d0 = np.floor(from_homogeneous(np.matmul(to_homogeneous(pts2d[img_idx0]), K.T))).astype(np.int)
    # print(pts2d0)

    new_img = np.zeros_like(imgf0)
    new_img[pts2d0[:, 1], pts2d0[:, 0]] = 1
    new_img = gaussian_filter(new_img, sigma=10) 
    new_img /= new_img.max()
    # print(new_img.shape)
    return torch.from_numpy(new_img.transpose((2,0,1)) * 255)

if __name__ == "__main__":
    model = FeaturePnP(iterations=100, device=torch.device('cuda:0'), loss_fn=squared_loss, lambda_=0.01, verbose=True)

    poses = scipy.io.loadmat('data/checkerboard/poses.mat')['poses']
    poses = np.concatenate((poses, np.zeros((210, 1, 4))), axis=1)
    poses[:, 3, 3] = 1
    depths = scipy.io.loadmat('data/checkerboard/depths.mat')['depths']
    pts2d = scipy.io.loadmat('data/checkerboard/pts2d.mat')['pts2d']
    pts3d = scipy.io.loadmat('data/checkerboard/pts3d.mat')['p_W_corners']

    img_idx0 = np.random.randint(len(poses))
    img_idx1 = np.random.randint(len(poses))
    assert img_idx0 != img_idx1

    pts0 = pts2d[img_idx0]
    pts1 = pts2d[img_idx1]
    z0_gt = depths[img_idx0]
    relative_pose = np.matmul(poses[img_idx1], np.linalg.inv(poses[img_idx0]))
    R_gt = torch.from_numpy(relative_pose[:3, :3]).type(torch.float32)
    t_gt = torch.from_numpy(relative_pose[:3, 3]).type(torch.float32)
            

    K = np.array([[420.506712, 0.        ,  355.208298],
                  [0.        , 420.610940,  250.336787],
                  [0.        , 0.        ,  1.]])

    imgf0 = keypoints2example(img_idx0)
    imgf1 = keypoints2example(img_idx1)
    initial_poses = np.loadtxt("data/checkerboard/initial_pose_my.txt")
    init_pose0 = initial_poses[img_idx0]
    init_pose1 = initial_poses[img_idx1]
    init_R0 = np.reshape(init_pose0[:9], (3,3))
    init_t0 = init_pose0[9:].reshape((3,1)) * 0.01 # 0.01 is the real scale for the data
    init_R1 = np.reshape(init_pose1[:9], (3,3))
    init_t1 = init_pose1[9:].reshape((3,1)) * 0.01 # 0.01 is the real scale for the data
    proj_0 = np.concatenate([init_R0, init_t0], axis=1)
    proj_0 = np.concatenate([proj_0, np.zeros((1,4))], axis=0)
    proj_0[3, 3] = 1
    proj_1 = np.concatenate([init_R1, init_t1], axis=1)
    proj_1 = np.concatenate([proj_1, np.zeros((1,4))], axis=0)
    proj_1[3, 3] = 1
    relative_pose_init = np.matmul(proj_1, np.linalg.inv(proj_0))

    R_init = torch.from_numpy(relative_pose_init[:3, :3]).type(torch.float32)
    t_init = torch.from_numpy(relative_pose_init[:3, 3]).type(torch.float32)
    

    dr = so3exp_map(torch.from_numpy(np.random.randn(3)).type(torch.float32) * 0.01)
    dt = torch.from_numpy(np.random.randn(3)).type(torch.float32) * 0.01
    proj_1_rnd = np.zeros((4, 4))
    proj_1_rnd[3, 3] = 1
    proj_1_rnd[:3, :3] = dr.numpy() @ proj_1[:3, :3]
    proj_1_rnd[:3, 3] = dr.numpy() @ proj_1[:3, 3] + dt.numpy()
    R_init_rnd = dr @ R_gt
    t_init_rnd = dr @ t_gt + dt
    # R_opt_init, t_opt_init = model(pts0, R_init, t_init, imgf0, imgf1, img1gx, img1gy, K, K, scale=scale, z0_gt=z0_gt, R_gt=R_gt, t_gt=t_gt, size_ratio=1)
    # R_opt_rnd, t_opt_rnd = model(pts0, R_init_rnd, t_init_rnd, imgf0, imgf1, img1gx, img1gy, K, K, scale=scale, z0_gt=z0_gt, R_gt=R_gt, t_gt=t_gt, size_ratio=1)

    # query_prediction = {'matrix': poses[img_idx1]}
    # reference_prediction = {'matrix': poses[img_idx0]}
    reference_prediction = {'matrix': proj_0}
    query_prediction = {'matrix': proj_1}
    local_reconstruction = {
        'intrinsics': K,
        'points_2D': np.matmul(to_homogeneous(pts0), K.T),
        # 'points_3D': np.matmul(to_homogeneous(to_homogeneous(pts0) * z0_gt[:, None]), np.linalg.inv(poses[img_idx0]).T),
        'points_3D': pts3d
    }
    query_dense_hypercolumn = imgf1.type(torch.float32).cuda()
    reference_dense_hypercolumn = imgf2.to(imgf1)
    query_intrinsics = K
    size_ratio = 1
    mask = np.ones((len(pts0)))

    R_opt_init, t_opt_init = model(
                dotdict(query_prediction), 
                dotdict(reference_prediction), 
                dotdict(local_reconstruction), 
                mask,
                query_dense_hypercolumn, 
                reference_dense_hypercolumn,
                query_intrinsics, 
                size_ratio,
                R_gt=R_gt, 
                t_gt=t_gt)

    
    query_prediction = {'matrix': proj_1_rnd}
    R_opt_rnd, t_opt_rnd = model(
                dotdict(query_prediction), 
                dotdict(reference_prediction), 
                dotdict(local_reconstruction), 
                mask,
                query_dense_hypercolumn, 
                reference_dense_hypercolumn,
                query_intrinsics, 
                size_ratio,
                R_gt=R_gt, 
                t_gt=t_gt)

    print("Initial pose error")
    print(pose_error(R_init, t_init, R_gt, t_gt))
    print("Optimized initial pose error")
    print(pose_error(R_opt_init, t_opt_init, R_gt, t_gt))
    print("Random perturbed pose error")
    print(pose_error(R_init_rnd, t_init_rnd, R_gt, t_gt))
    print("optimized random pose error")
    print(pose_error(R_opt_rnd, t_opt_rnd, R_gt, t_gt))