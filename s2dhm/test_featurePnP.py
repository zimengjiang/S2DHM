import torch
import numpy as np
from torch import nn

from featurePnP.utils import (to_homogeneous, from_homogeneous, batched_eye_like,
                    skew_symmetric, so3exp_map, sobel_filter)
# from featurePnP.losses import scaled_loss, squared_loss, pose_error, pose_error_np
from featurePnP.losses import *
from featurePnP.optimization import FeaturePnP
import scipy
import imageio
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from scipy import io
from visualization import plot_correspondences
from pose_prediction import keypoint_association
import gin
from network import network
import os
from pose_prediction import predictor
from datasets import base_dataset
from datasets.cmu_dataset import ExtendedCMUDataset
from pose_prediction.sparse_to_dense_predictor import SparseToDensePredictor
from pose_prediction.matrix_utils import quaternion_matrix
from image_retrieval import rank_images
from pose_prediction import exhaustive_search
from PIL import Image
from pose_prediction import solve_pnp
from tqdm import tqdm
from functools import partial

@gin.configurable
def get_dataset_loader(dataset_loader_cls):
    return dataset_loader_cls

@gin.configurable
def get_pose_predictor(pose_predictor_cls: predictor.PosePredictor,
                       dataset: base_dataset.BaseDataset,
                       network: network.ImageRetrievalModel,
                       ranks: np.ndarray,
                       log_images: bool):
    return pose_predictor_cls(dataset=dataset,
                              network=network,
                              ranks=ranks,
                              log_images=log_images)

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

    imgf0 = imageio.imread("data/checkerboard/images_undistorted/img_{:04d}.jpg".format(img_idx0+1)).astype(np.float32)
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
    return torch.from_numpy(new_img.transpose((2,0,1)) * 255)[None, ...]


def test_toy_example():
    model = FeaturePnP(iterations=50, device=torch.device('cuda:0'), loss_fn=squared_loss, init_lambda=0.1, verbose=True)

    poses = scipy.io.loadmat('data/checkerboard/poses.mat')['poses']
    poses = np.concatenate((poses, np.zeros((210, 1, 4))), axis=1)
    poses[:, 3, 3] = 1
    depths = scipy.io.loadmat('data/checkerboard/depths.mat')['depths']
    pts2d = scipy.io.loadmat('data/checkerboard/pts2d.mat')['pts2d']
    pts3d = scipy.io.loadmat('data/checkerboard/pts3d.mat')['p_W_corners']

    # img_idx0 = np.random.randint(len(poses))
    # img_idx1 = np.random.randint(len(poses))
    img_idx0 = 0
    img_idx1 = 195
    assert img_idx0 != img_idx1

    pts0 = pts2d[img_idx0]
    pts1 = pts2d[img_idx1]
    z0_gt = depths[img_idx0]
    pts3d0 = to_homogeneous(pts0) * z0_gt[..., None]
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

    # print(from_homogeneous(to_homogeneous(pts3d0) @ np.linalg.inv(proj_0.T) @ proj_0.T))
    # print(pts3d)
    pts3d = from_homogeneous(to_homogeneous(pts3d0) @ np.linalg.inv(proj_0.T))

    R_init = torch.from_numpy(relative_pose_init[:3, :3]).type(torch.float32)
    t_init = torch.from_numpy(relative_pose_init[:3, 3]).type(torch.float32)
    

    rnd1 = np.array([0.24581696, 0.42776546, 0.00600544])
    rnd2 = np.array([ 0.62033623, -0.25213616, -0.54241641])
    dr = so3exp_map(torch.from_numpy(rnd1).type(torch.float32) * 0.01)
    dt = torch.from_numpy(rnd2).type(torch.float32) * 0.01
    # dr = so3exp_map(torch.from_numpy(np.random.randn(3)).type(torch.float32) * 0.01)
    # dt = torch.from_numpy(np.random.randn(3)).type(torch.float32) * 0.01
    proj_1_rnd = np.zeros((4, 4))
    proj_1_rnd[3, 3] = 1
    proj_1_rnd[:3, :3] = dr.numpy() @ proj_1[:3, :3]
    proj_1_rnd[:3, 3] = dr.numpy() @ proj_1[:3, 3] + dt.numpy()
    # print(proj_1)
    # print(proj_1_rnd)
    R_init_rnd = dr @ R_init
    t_init_rnd = dr @ t_init + dt
    # R_opt_init, t_opt_init = model(pts0, R_init, t_init, imgf0, imgf1, img1gx, img1gy, K, K, scale=scale, z0_gt=z0_gt, R_gt=R_gt, t_gt=t_gt, size_ratio=1)
    # R_opt_rnd, t_opt_rnd = model(pts0, R_init_rnd, t_init_rnd, imgf0, imgf1, img1gx, img1gy, K, K, scale=scale, z0_gt=z0_gt, R_gt=R_gt, t_gt=t_gt, size_ratio=1)

    # query_prediction = {'matrix': poses[img_idx1]}
    # reference_prediction = {'matrix': poses[img_idx0]}
    # reference_prediction = {'matrix': proj_0}
    # query_prediction = {'matrix': proj_1}
    reference_prediction = {'matrix': proj_0}
    query_prediction = {'matrix': proj_1}
    local_reconstruction = {
        'intrinsics': K,
        'points_2D': np.matmul(to_homogeneous(pts0), K.T),
        # 'points_3D': np.matmul(to_homogeneous(to_homogeneous(pts0) * z0_gt[:, None]), np.linalg.inv(poses[img_idx0]).T),
        'points_3D': pts3d
    }
    query_dense_hypercolumn = imgf1.type(torch.float32).cuda()
    reference_dense_hypercolumn = imgf0.type(torch.float32).cuda()
    query_intrinsics = K
    size_ratio = 1
    mask = np.ones((len(pts0)), dtype=np.bool)

    R_opt_init, t_opt_init = model(
                dotdict(query_prediction), 
                dotdict(reference_prediction), 
                dotdict(local_reconstruction), 
                mask,
                query_dense_hypercolumn, 
                reference_dense_hypercolumn,
                query_intrinsics, 
                size_ratio,
                R_gt=R_gt.cuda(), 
                t_gt=t_gt.cuda())

    
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
                R_gt=R_gt.cuda(), 
                t_gt=t_gt.cuda())

    R_opt_init = R_opt_init.cpu()
    t_opt_init = t_opt_init.cpu()
    R_opt_rnd = R_opt_rnd.cpu()
    t_opt_rnd = t_opt_rnd.cpu()
    print("Initial pose error")
    print(pose_error(R_init, t_init, R_gt, t_gt))
    print("Optimized initial pose error")
    print(pose_error(R_opt_init, t_opt_init, R_gt, t_gt))
    print("Random perturbed pose error")
    print(pose_error(R_init_rnd, t_init_rnd, R_gt, t_gt))
    print("optimized random pose error")
    print(pose_error(R_opt_rnd, t_opt_rnd, R_gt, t_gt))

def bind_cmu_parameters(cmu_slice, mode):
    """Update CMU gin parameters to match chosen slice."""
    gin.bind_parameter('ExtendedCMUDataset.cmu_slice', cmu_slice)
    if mode=='nearest_neighbor':
        gin.bind_parameter('NearestNeighborPredictor.output_filename',
            '../results/cmu/slice_{}/top_1_predictions.txt'.format(cmu_slice))
    elif mode=='superpoint':
        gin.bind_parameter('SuperPointPredictor.output_filename',
            '../results/cmu/slice_{}/superpoint_predictions.txt'.format(cmu_slice))
        gin.bind_parameter('plot_correspondences.plot_correspondences.export_folder',
            '../logs/superpoint/correspondences/cmu/slice_{}/'.format(cmu_slice))
        gin.bind_parameter('plot_correspondences.plot_detections.export_folder',
            '../logs/superpoint/detections/cmu/slice_{}/'.format(cmu_slice))
        gin.bind_parameter('plot_correspondences.plot_image_retrieval.export_folder',
            '../logs/superpoint/nearest_neighbor/cmu/slice_{}/'.format(cmu_slice))
    elif mode=='sparse_to_dense':
        gin.bind_parameter('SparseToDensePredictor.output_filename',
            '../results/cmu/slice_{}/sparse_to_dense_predictions.txt'.format(cmu_slice))
        gin.bind_parameter('plot_correspondences.plot_correspondences.export_folder',
            '../logs/sparse_to_dense/correspondences/cmu/slice_{}/'.format(cmu_slice))
        gin.bind_parameter('plot_correspondences.plot_image_retrieval.export_folder',
            '../logs/sparse_to_dense/nearest_neighbor/cmu/slice_{}/'.format(cmu_slice)),

def test_cmu_images():
    cmu_root = "/local/home/lixxue/S2DHM/data/cmu_extended/"
    slice_idx = 2
    bind_cmu_parameters(2, 'sparse_to_dense') 
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    gin.parse_config_file("configs/runs/run_sparse_to_dense_on_cmu.gin")
    dataset = get_dataset_loader()
    # query_image = dataset.data['reference_image_names'][4]
    # print(query_image)
    # dataset.data['query_image_names'] = [query_image]
    dataset.data['query_image_names'] = dataset.data['reference_image_names']
    net = network.ImageRetrievalModel(device=device)
    ranks = rank_images.fetch_or_compute_ranks(dataset, net).T

    pose_predictor = get_pose_predictor(dataset=dataset,
                                        network=net,
                                        ranks=ranks,
                                        log_images=True)
    # print(dataset.data['query_image_names'])
    num_images = len(dataset.data['query_image_names'])
    top_N = 5
    init_pose_error = np.zeros((num_images, top_N, 2))
    init_num_inliers = np.zeros((num_images, top_N))
    fpnp_pose_error = np.zeros((num_images, top_N, 2))
    fpnp_num_inliers = np.zeros((num_images, top_N))

    # fPnP = FeaturePnP(iterations=1000, device=torch.device('cuda'), loss_fn=squared_loss, init_lambda=0.01, verbose=False)
    # loss_fn = squared_loss
    # out_fname = "results/sqloss_cmu_slice_2"
    # loss_fn = partial(barron_loss, alpha=torch.zeros((1)).cuda()) # cauchy loss 
    # out_fname = "results/cauchy_loss_cmu_slice_2"
    loss_fn = partial(barron_loss, alpha=-2*torch.ones((1)).cuda()) # cauchy loss 
    out_fname = "results/gm_loss_cmu_slice_2"
    # loss_fn = huber_loss 
    # out_fname = "results/huber_loss_cmu_slice_2"
    fPnP = FeaturePnP(iterations=1000, device=torch.device('cuda'), loss_fn=loss_fn, init_lambda=0.01, verbose=False)
    for i, query_image in tqdm(enumerate(dataset.data['query_image_names']), total=num_images):
        query_idx = dataset.data['query_image_names'].index(query_image)
        query_dense_hypercolumn, _ = pose_predictor._network.compute_hypercolumn([query_image], to_cpu=False, resize=True)
        channels, width, height = query_dense_hypercolumn.shape[1:]
        query_dense_hypercolumn_copy = query_dense_hypercolumn.clone().detach()
        query_dense_hypercolumn = query_dense_hypercolumn.squeeze().view(
            (channels, -1))

        local_reconstruction = \
            dataset.data['filename_to_local_reconstruction'][query_image]
        if local_reconstruction.intrinsics is None:
            continue
        ground_truth = solve_pnp.solve_pnp(
            points_2D=local_reconstruction.points_2D,
            points_3D=local_reconstruction.points_3D,
            intrinsics=local_reconstruction.intrinsics,
            distortion_coefficients=local_reconstruction.distortion_coefficients,
            reference_filename=None,
            reference_2D_points=local_reconstruction.points_2D,
            reference_keypoints=None,
        )
        R_gt = torch.from_numpy(ground_truth.matrix[:3, :3].astype(np.float32)).cuda()
        t_gt = torch.from_numpy(ground_truth.matrix[:3, 3].astype(np.float32)).cuda()

        for j in range(top_N+1):
            nn_idx = ranks[query_idx][j]
            nearest_neighbor = dataset.data['reference_image_names'][nn_idx]
            if nearest_neighbor == query_image:
                assert j == 0
                # skip the same image
                # print("find exact the same image at rank {}".format(j))
                continue
            local_reconstruction = \
                dataset.data['filename_to_local_reconstruction'][nearest_neighbor]
            reference_sparse_hypercolumns, cell_size, reference_dense_hypercolumn = \
                pose_predictor._compute_sparse_reference_hypercolumn(
                nearest_neighbor, local_reconstruction, return_dense=True)
            reference_prediction = pose_predictor._nearest_neighbor_prediction(
                nearest_neighbor)


            matches_2D, mask = exhaustive_search.exhaustive_search(
                query_dense_hypercolumn,
                reference_sparse_hypercolumns,
                Image.open(nearest_neighbor).size[::-1],
                [width, height],
                cell_size)

            points_2D = np.reshape(
                matches_2D.cpu().numpy()[mask], (-1, 1, 2))
            points_3D = np.reshape(
                local_reconstruction.points_3D[mask], (-1, 1, 3))
            distortion_coefficients = \
                local_reconstruction.distortion_coefficients
    
            # query_intrinsics, query_distortion_coefficients = pose_predictor._filename_to_intrinsics[query_image] 
            prediction = solve_pnp.solve_pnp(
                points_2D=points_2D,
                points_3D=points_3D,
                intrinsics=local_reconstruction.intrinsics,
                distortion_coefficients=local_reconstruction.distortion_coefficients,
                # intrinsics=query_intrinsics,
                # distortion_coefficients=query_distortion_coefficients,
                reference_filename=nearest_neighbor,
                reference_2D_points=local_reconstruction.points_2D[mask],
                reference_keypoints=None)

            if not prediction.success:
                continue

            # print(prediction.num_inliers)
            points_3D_proj = from_homogeneous(from_homogeneous(to_homogeneous(points_3D)  @ prediction.matrix.T) @ local_reconstruction.intrinsics.T)
            dist = torch.sum(torch.pow(torch.Tensor(points_2D - points_3D_proj), 2), dim=-1)
            # print(torch.sum(dist < 12*12))
            init_num_inliers[i, j-1] = torch.sum(dist < 12*12).item()

            R_pred = torch.from_numpy(prediction.matrix[:3, :3].astype(np.float32)).cuda()
            t_pred = torch.from_numpy(prediction.matrix[:3, 3].astype(np.float32)).cuda()

            # print(t_gt, t_pred)
            init_R_error, init_t_error = pose_error(R_gt, t_gt, R_pred, t_pred)
            # print("initial pose error: ", init_R_error.item(), init_t_error.item())
            init_pose_error[i, j-1] = init_R_error.item(), init_t_error.item()

            fPnP(
                query_prediction=prediction, 
                reference_prediction=reference_prediction, 
                local_reconstruction=local_reconstruction,
                mask=mask,
                query_dense_hypercolumn=query_dense_hypercolumn_copy, 
                reference_dense_hypercolumn=reference_dense_hypercolumn,
                query_intrinsics=local_reconstruction.intrinsics,
                size_ratio=cell_size[0], 
                points_2D=matches_2D[mask].cuda(),
                R_gt=R_gt, 
                t_gt=t_gt,
                )

            fpnp_num_inliers[i, j-1] = prediction.num_inliers
            R_pred = torch.from_numpy(prediction.matrix[:3, :3].astype(np.float32)).cuda()
            t_pred = torch.from_numpy(prediction.matrix[:3, 3].astype(np.float32)).cuda()
            fpnp_R_error, fpnp_t_error = pose_error(R_gt, t_gt, R_pred, t_pred)
            fpnp_pose_error[i, j-1] = fpnp_R_error.item(), fpnp_t_error.item()

            # print("initial pose error: {:.3f} {:.3f}".format(init_pose_error[i, j-1, 0], init_pose_error[i, j-1, 1]))
            # print("initial num inliers: {:.0f} / {:.0f}".format(init_num_inliers[i, j-1], len(points_3D)))
            # print("fpnp pose error: {:.3f} {:.3f}".format(fpnp_pose_error[i, j-1, 0], fpnp_pose_error[i, j-1, 1]))
            # print("fpnp num inliers: {:.0f} / {:.0f}".format(fpnp_num_inliers[i, j-1], len(points_3D)))

    out_dict = {
        'init_num_inliers': init_num_inliers,
        'init_pose_error': init_pose_error,
        'fpnp_num_inliers': fpnp_num_inliers,
        'fpnp_pose_error': fpnp_pose_error,
    }
    np.save(out_fname, out_dict) 
    return

    
if __name__ == "__main__":
    test_cmu_images()