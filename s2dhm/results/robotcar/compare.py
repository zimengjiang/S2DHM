import os
import numpy as np
from matplotlib import pyplot as plt

fnames = [fname for fname in os.listdir("./") if ".npy" in fname]
print(fnames)

for fname in fnames:
    array_dict = np.load(fname)[()]
    for key in array_dict.keys():
        # print(key, array_dict[key].shape)
        num_images = array_dict[key].shape[0]
        if "inliers" in key:
            best_idxs = np.argmax(array_dict[key], axis=1)
            # print(best_idxs)
            gt = np.amax(array_dict[key], axis=1)
            # print(np.amax(array_dict[key], axis=1))
            myresult = array_dict[key][np.arange(num_images), best_idxs]
            # print(myresult[:, None].shape)
            # print(np.all(gt == myresult))
        else:
            # gt = np.amax(array_dict[key], axis=1)
            # print(np.amax(array_dict[key], axis=1))
            myresult = array_dict[key][np.arange(num_images), best_idxs, :][:, None, :]
            # print(myresult.shape)
    break

def choose_best_prediction(array_dict):
    num_images = array_dict['init_num_inliers'].shape[0]
    # k = 5
    new_array_dict = {}
    best_init_idx = np.argmax(array_dict['init_num_inliers'], axis=1)
    new_array_dict['init_num_inliers'] = array_dict['init_num_inliers'][np.arange(num_images), best_idxs][:, None]
    new_array_dict['init_pose_error'] = array_dict['init_pose_error'][np.arange(num_images), best_idxs][:, None, :]
    new_array_dict['fpnp_num_inliers'] = array_dict['fpnp_num_inliers'][np.arange(num_images), best_idxs][:, None]
    new_array_dict['fpnp_pose_error'] = array_dict['fpnp_pose_error'][np.arange(num_images), best_idxs][:, None, :]
    return new_array_dict
    # best_init_inliers = np.zeros((num_images, 1))
    # best_init_pose_error = np.zeros((num_images, 1, 2))
    # best_init_inliers = np.zeros((num_images, 1))
    # best_init_pose_error = np.zeros((num_images, 1, 2))

def calculate_stats(array_dict):
    stats = {}
    array_dict = choose_best_prediction(array_dict)
    rPnP_inliers = array_dict['init_num_inliers']
    fPnP_inliers = array_dict['fpnp_num_inliers']
    rPnP_perror = array_dict['init_pose_error']
    fPnP_perror = array_dict['fpnp_pose_error']
    rPnP_rerror = array_dict['init_pose_error'][:, :, 0]
    fPnP_rerror = array_dict['fpnp_pose_error'][:, :, 0]
    rPnP_terror = array_dict['init_pose_error'][:, :, 1]
    fPnP_terror = array_dict['fpnp_pose_error'][:, :, 1]

    rPnP_high_prec = np.logical_and(rPnP_rerror < 2, rPnP_terror < 0.25)
    fPnP_high_prec = np.logical_and(fPnP_rerror < 2, fPnP_terror < 0.25)
    rPnP_medium_prec = np.logical_and(rPnP_rerror < 5, rPnP_terror < 0.5)
    fPnP_medium_prec = np.logical_and(fPnP_rerror < 5, fPnP_terror < 0.5)
    rPnP_coarse_prec = np.logical_and(rPnP_rerror < 10, rPnP_terror < 5)
    fPnP_coarse_prec = np.logical_and(fPnP_rerror < 10, fPnP_terror < 5)

    stats['mean_rPnP_num_inliers'] = np.mean(array_dict['init_num_inliers'])
    stats['mean_rPnP_rerror'] = np.mean(array_dict['init_pose_error'][:, :, 0])
    stats['mean_rPnP_terror'] = np.mean(array_dict['init_pose_error'][:, :, 1])
    stats['mean_fPnP_num_inliers'] = np.mean(array_dict['fpnp_num_inliers'])
    stats['mean_fPnP_rerror'] = np.mean(array_dict['fpnp_pose_error'][:, :, 0])
    stats['mean_fPnP_terror'] = np.mean(array_dict['fpnp_pose_error'][:, :, 1])
    # stats['mean_top1_rPnP_num_inliers'] = np.mean(rPnP_inliers, axis=0)[0]
    # stats['mean_top1_fPnP_num_inliers'] = np.mean(fPnP_inliers, axis=0)[0]
    # stats['mean_top1_rPnP_pose_error'] = np.mean(rPnP_perror, axis=0)[0] 
    # stats['mean_top1_fPnP_pose_error'] = np.mean(fPnP_perror, axis=0)[0]
    stats['rPnP_high_prec'] = np.mean(rPnP_high_prec, axis=0)[0]
    stats['rPnP_medium_prec'] = np.mean(rPnP_medium_prec, axis=0)[0]
    stats['rPnP_coarse_prec'] = np.mean(rPnP_coarse_prec, axis=0)[0]
    stats['fPnP_high_prec'] = np.mean(fPnP_high_prec, axis=0)[0]
    stats['fPnP_medium_prec'] = np.mean(fPnP_medium_prec, axis=0)[0]
    stats['fPnP_coarse_prec'] = np.mean(fPnP_coarse_prec, axis=0)[0]
    for key in stats.keys():
        print("{}: {:.3f}".format(key, stats[key]))
    return stats

for fname in fnames:
    array_dict = np.load(fname)[()]
    print(fname)
    calculate_stats(array_dict)
