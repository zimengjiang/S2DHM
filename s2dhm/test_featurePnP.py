import torch
import numpy as np
from torch import nn
import scipy
from scipy import io
import imageio
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
import gin
import os
from PIL import Image
from tqdm import tqdm
from functools import partial
import argparse

from featurePnP.utils import to_homogeneous, from_homogeneous, batched_eye_like, skew_symmetric, so3exp_map, sobel_filter, np_gradient_filter
from featurePnP.losses import scaled_loss, squared_loss, huber_loss, barron_loss, pose_error, pose_error_np, pose_error_mat
from featurePnP.optimization import FeaturePnP
from visualization import plot_correspondences
from network import network
from datasets import base_dataset
from datasets.cmu_dataset import ExtendedCMUDataset
from pose_prediction import predictor, keypoint_association, exhaustive_search, solve_pnp
from pose_prediction.sparse_to_dense_predictor import SparseToDensePredictor
from pose_prediction.matrix_utils import quaternion_matrix
from image_retrieval import rank_images


# used for create struct for fPnP
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

# test on toy example for verification of fPnP
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


# helper functions from run.py
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

# test on a subset of reference images in robotcar for local parameter tuning
def test_on_real_images(args):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    if args.dataset == 'robotcar':
        gin.parse_config_file("configs/runs/run_sparse_to_dense_on_robotcar.gin")
    else:
        # for now we only test on slice 2
        slice_idx = 2
        bind_cmu_parameters(slice_idx, 'sparse_to_dense') 
        gin.parse_config_file("configs/runs/run_sparse_to_dense_on_cmu.gin")
    dataset = get_dataset_loader()
    num_ref_images = len(dataset.data['reference_image_names'])
    if args.dataset == 'robotcar':
        # a random run of np.random.choice(num_images, 500, replace=False)
        idxs = np.array([4, 79, 84, 86, 99, 103, 108, 124, 132, 139, 150, 166, 171, 172, 176, 182, 198, 206, 211, 232, 240, 257, 262, 269, 279, 291, 295, 303, 315, 321, 329, 349, 363, 380, 384, 400, 419, 430, 451, 472, 476, 480, 495, 499, 558, 589, 599, 615, 626, 671, 674, 684, 685, 707, 712, 714, 726, 735, 759, 813, 859, 863, 878, 888, 898, 899, 906, 912, 921, 927, 938, 963, 980, 990, 996, 1025, 1037, 1043, 1053, 1064, 1068, 1070, 1075, 1097, 1122, 1147, 1160, 1174, 1204, 1245, 1247, 1255, 1261, 1262, 1279, 1282, 1287, 1303, 1337, 1356, 1400, 1405, 1406, 1408, 1414, 1417, 1420, 1431, 1460, 1464, 1465, 1468, 1477, 1480, 1596, 1606, 1608, 1635, 1636, 1641, 1645, 1649, 1686, 1701, 1720, 1733, 1792, 1793, 1794, 1801, 1814, 1833, 1835, 1839, 1847, 1854, 1857, 1886, 1888, 1892, 1910, 1931, 1934, 1971, 1986, 1987, 2000, 2007, 2008, 2046, 2052, 2054, 2076, 2087, 2112, 2124, 2132, 2140, 2163, 2165, 2200, 2220, 2242, 2283, 2290, 2300, 2325, 2342, 2358, 2372, 2396, 2417, 2419, 2457, 2458, 2472, 2476, 2488, 2495, 2508, 2527, 2533, 2549, 2550, 2568, 2590, 2609, 2613, 2625, 2637, 2646, 2677, 2685, 2693, 2702, 2704, 2715, 2716, 2718, 2723, 2724, 2727, 2731, 2750, 2760, 2761, 2770, 2782, 2786, 2801, 2813, 2818, 2828, 2830, 2845, 2846, 2885, 2886, 2892, 2907, 2912, 2914, 2925, 2935, 2947, 2962, 2973, 2986, 2997, 3033, 3040, 3054, 3061, 3094, 3102, 3110, 3114, 3116, 3136, 3158, 3162, 3193, 3218, 3230, 3241, 3250, 3275, 3293, 3297, 3300, 3306, 3318, 3342, 3427, 3435, 3439, 3460, 3475, 3497, 3506, 3557, 3563, 3594, 3604, 3618, 3672, 3675, 3708, 3738, 3766, 3773, 3783, 3786, 3825, 3831, 3839, 3842, 3852, 3887, 3898, 3908, 3915, 3916, 3927, 3937, 3940, 3950, 3959, 3983, 3990, 4003, 4021, 4030, 4036, 4042, 4044, 4062, 4104, 4120, 4126, 4146, 4156, 4230, 4262, 4267, 4279, 4281, 4289, 4290, 4302, 4315, 4337, 4376, 4411, 4416, 4418, 4424, 4430, 4435, 4465, 4512, 4519, 4527, 4535, 4544, 4547, 4551, 4570, 4573, 4583, 4630, 4631, 4644, 4653, 4669, 4671, 4682, 4693, 4703, 4724, 4739, 4743, 4745, 4748, 4808, 4823, 4879, 4894, 4903, 4913, 4921, 4926, 4948, 4953, 4961, 4992, 5003, 5011, 5023, 5038, 5048, 5080, 5081, 5090, 5091, 5095, 5103, 5134, 5142, 5147, 5157, 5168, 5192, 5211, 5219, 5226, 5227, 5238, 5250, 5251, 5252, 5268, 5331, 5373, 5380, 5392, 5395, 5396, 5410, 5437, 5447, 5448, 5528, 5540, 5553, 5564, 5572, 5581, 5584, 5590, 5595, 5596, 5600, 5609, 5614, 5658, 5675, 5682, 5691, 5723, 5747, 5760, 5779, 5815, 5816, 5818, 5836, 5890, 5894, 5949, 5953, 5964, 5968, 5978, 5988, 5992, 6000, 6005, 6008, 6009, 6027, 6032, 6035, 6052, 6071, 6073, 6098, 6155, 6160, 6170, 6171, 6176, 6185, 6188, 6189, 6190, 6205, 6227, 6245, 6253, 6255, 6261, 6271, 6281, 6323, 6326, 6348, 6379, 6383, 6387, 6394, 6404, 6436, 6437, 6455, 6467, 6506, 6513, 6518, 6520, 6521, 6535, 6537, 6542, 6553, 6557, 6564, 6593, 6595, 6611, 6636, 6667, 6690, 6702, 6714, 6736, 6743, 6796, 6801, 6806, 6811, 6817, 6818, 6862, 6885, 6903, 6916, 6943, 6949, 6952])
        # idxs = np.arange(50)
    else:
        idxs = np.arange(len(num_ref_images))
    # random select 500 images for local validation
    # np.random.seed(0)
    # idxs = np.random.choice(num_ref_images, min(500, num_ref_images), replace=False)
    dataset.data['query_image_names'] = [dataset.data['reference_image_names'][idx] for idx in idxs]
    net = network.ImageRetrievalModel(device=device, vgg16fmap_ckpt=args.ckpt)
    ranks = rank_images.fetch_or_compute_ranks(dataset, net).T

    pose_predictor = get_pose_predictor(dataset=dataset,
                                        network=net,
                                        ranks=ranks,
                                        log_images=True)
    num_images = len(dataset.data['query_image_names'])
    top_N = 30
    init_pose_error = np.zeros((num_images, top_N, 2))
    init_num_inliers = np.zeros((num_images, top_N))
    fpnp_pose_error = np.zeros((num_images, top_N, 2))
    fpnp_num_inliers = np.zeros((num_images, top_N))

    use_sobel = args.use_sobel
    if args.loss_fn == 'sq':
        loss_fn = squared_loss
    elif args.loss_fn == 'cauchy':
        loss_fn = partial(barron_loss, alpha=torch.zeros((1)).cuda()) # cauchy loss 
    elif args.loss_fn == 'gm':
        loss_fn = partial(barron_loss, alpha=-2*torch.ones((1)).cuda()) #  Geman-McClure loss 
    elif args.loss_fn == 'huber':
        loss_fn = huber_loss 
    os.makedirs(os.path.join("results", args.dataset), exist_ok=True)
    out_fname = os.path.join("results", args.dataset, args.out_fname)
    fPnP = FeaturePnP(iterations=1000, device=torch.device('cuda'), loss_fn=loss_fn, init_lambda=0.01, verbose=False)
    for i, query_image in tqdm(enumerate(dataset.data['query_image_names']), total=num_images):
        query_idx = dataset.data['query_image_names'].index(query_image)
        query_dense_hypercolumn, _ = pose_predictor._network.compute_hypercolumn([query_image], to_cpu=False, resize=True)
        channels, width, height = query_dense_hypercolumn.shape[1:]
        query_dense_hypercolumn_copy = query_dense_hypercolumn.clone().detach()
        query_dense_hypercolumn = query_dense_hypercolumn.squeeze().view((channels, -1))
        print(query_dense_hypercolumn_copy.shape)

        query_local_reconstruction = dataset.data['filename_to_local_reconstruction'][query_image]
        ground_truth = solve_pnp.solve_pnp(
            points_2D=query_local_reconstruction.points_2D,
            points_3D=query_local_reconstruction.points_3D,
            intrinsics=query_local_reconstruction.intrinsics,
            distortion_coefficients=query_local_reconstruction.distortion_coefficients,
            reference_filename=None,
            reference_2D_points=query_local_reconstruction.points_2D,
            reference_keypoints=None,
        )
        R_gt = torch.from_numpy(ground_truth.matrix[:3, :3].astype(np.float32)).cuda()
        t_gt = torch.from_numpy(ground_truth.matrix[:3, 3].astype(np.float32)).cuda()

        for j in range(top_N+1):
            nn_idx = ranks[query_idx][j]
            nearest_neighbor = dataset.data['reference_image_names'][nn_idx]
            # skip the same image
            if nearest_neighbor == query_image:
                assert j == 0
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
    
            query_intrinsics = query_local_reconstruction.intrinsics 
            query_distortion_coefficients = query_local_reconstruction.distortion_coefficients
            prediction = solve_pnp.solve_pnp(
                points_2D=points_2D,
                points_3D=points_3D,
                # intrinsics=local_reconstruction.intrinsics,
                # distortion_coefficients=local_reconstruction.distortion_coefficients,
                intrinsics=query_intrinsics,
                distortion_coefficients=query_distortion_coefficients,
                reference_filename=nearest_neighbor,
                reference_2D_points=local_reconstruction.points_2D[mask],
                reference_keypoints=None)

            if not prediction.success:
                continue

            points_3D_proj = from_homogeneous(from_homogeneous(to_homogeneous(points_3D)  @ prediction.matrix.T) @ local_reconstruction.intrinsics.T)
            dist = torch.sum(torch.pow(torch.Tensor(points_2D - points_3D_proj), 2), dim=-1)
            init_num_inliers[i, j-1] = torch.sum(dist < 12*12).item()

            R_pred = torch.from_numpy(prediction.matrix[:3, :3].astype(np.float32)).cuda()
            t_pred = torch.from_numpy(prediction.matrix[:3, 3].astype(np.float32)).cuda()

            init_R_error, init_t_error = pose_error(R_gt, t_gt, R_pred, t_pred)
            init_pose_error[i, j-1] = init_R_error.item(), init_t_error.item()

            fPnP(
                query_prediction=prediction, 
                reference_prediction=reference_prediction, 
                local_reconstruction=local_reconstruction,
                mask=mask,
                query_dense_hypercolumn=query_dense_hypercolumn_copy, 
                reference_dense_hypercolumn=reference_dense_hypercolumn,
                query_intrinsics=query_intrinsics,
                size_ratio=cell_size[0], 
                points_2D=matches_2D[mask].cuda(),
                R_gt=R_gt, 
                t_gt=t_gt,
                use_sobel=use_sobel,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=['cmu', 'robotcar'])
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--out_fname", type=str)
    parser.add_argument("--loss_fn", choices=['sq', 'huber', 'cauchy', 'gm'])
    parser.add_argument("--use_sobel", action='store_true')
    args = parser.parse_args()
    print(args)

    test_on_real_images(args)
