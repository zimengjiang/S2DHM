import os
import numpy as np
from matplotlib import pyplot as plt
import argparse
import csv

def choose_best_prediction(array_dict):
    num_images = array_dict['init_num_inliers'].shape[0]
    new_array_dict = {}
    best_init_idxs = np.argmax(array_dict['init_num_inliers'], axis=1)
    new_array_dict['init_num_inliers'] = array_dict['init_num_inliers'][np.arange(num_images), best_init_idxs][:, None]
    new_array_dict['init_pose_error'] = array_dict['init_pose_error'][np.arange(num_images), best_init_idxs][:, None, :]
    best_fpnp_idxs = np.argmax(array_dict['fpnp_num_inliers'], axis=1)
    new_array_dict['fpnp_num_inliers'] = array_dict['fpnp_num_inliers'][np.arange(num_images), best_fpnp_idxs][:, None]
    new_array_dict['fpnp_pose_error'] = array_dict['fpnp_pose_error'][np.arange(num_images), best_fpnp_idxs][:, None, :]
    return new_array_dict

stats_keys = ['experiment','mean_rPnP_num_inliers','mean_rPnP_rerror','mean_rPnP_terror','mean_fPnP_num_inliers','mean_fPnP_rerror','mean_fPnP_terror','rPnP_high_prec','fPnP_high_prec','rPnP_medium_prec','fPnP_medium_prec','rPnP_coarse_prec','fPnP_coarse_prec']

def calculate_stats(array_dict, fname):
    stats = {}
    array_dict = choose_best_prediction(array_dict)
    rPnP_inliers = array_dict['init_num_inliers']
    fPnP_inliers = array_dict['fpnp_num_inliers']
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

    stats['experiment'] = fname
    stats['mean_rPnP_num_inliers'] = np.mean(rPnP_inliers)
    stats['mean_fPnP_num_inliers'] = np.mean(fPnP_inliers)
    stats['mean_rPnP_rerror'] = np.mean(rPnP_rerror)
    stats['mean_fPnP_rerror'] = np.mean(fPnP_rerror)
    stats['mean_rPnP_terror'] = np.mean(rPnP_terror)
    stats['mean_fPnP_terror'] = np.mean(fPnP_terror)

    stats['rPnP_high_prec'] = np.mean(rPnP_high_prec, axis=0)[0]
    stats['fPnP_high_prec'] = np.mean(fPnP_high_prec, axis=0)[0]
    stats['rPnP_medium_prec'] = np.mean(rPnP_medium_prec, axis=0)[0]
    stats['fPnP_medium_prec'] = np.mean(fPnP_medium_prec, axis=0)[0]
    stats['rPnP_coarse_prec'] = np.mean(rPnP_coarse_prec, axis=0)[0]
    stats['fPnP_coarse_prec'] = np.mean(fPnP_coarse_prec, axis=0)[0]
    # for key in stats.keys():
    #     print("{}: {:.3f}".format(key, stats[key]))
    return stats


def write_to_csv(fname, all_stats):
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\\\\ \n', delimiter='&', quoting=csv.QUOTE_NONE, quotechar='', escapechar='\\')
        header = [key.replace('_', '') for key in stats_keys]
        print(header)
        writer.writerow(header)
        for stats in all_stats:
            row = []
            for key in stats_keys:
                if 'prec' in key:
                    row.append("{:.1f}".format(stats[key]*100))
                elif 'error' in key:
                    row.append("{:.3f}".format(stats[key]))
                elif 'num_inliers' in key:
                    row.append("{:.1f}".format(stats[key]))
                elif 'retrieval' in stats[key]:
                    row.append(stats[key][10:-4].replace('_', ' '))
                else:
                    row.append(stats[key])
                row[-1] += '\t'
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="./")
    parser.add_argument("-f", "--fname", type=str, default="results.csv")
    args = parser.parse_args()

    fnames = [fname for fname in os.listdir(args.path) if ".npy" in fname]
    print(fnames)

    # for fname in fnames:
    #     array_dict = np.load(fname)[()]
    #     for key in array_dict.keys():
    #         # print(key, array_dict[key].shape)
    #         num_images = array_dict[key].shape[0]
    #         if "inliers" in key:
    #             best_idxs = np.argmax(array_dict[key], axis=1)
    #             # print(best_idxs)
    #             gt = np.amax(array_dict[key], axis=1)
    #             # print(np.amax(array_dict[key], axis=1))
    #             myresult = array_dict[key][np.arange(num_images), best_idxs]
    #             # print(myresult[:, None].shape)
    #             # print(np.all(gt == myresult))
    #             print(np.mean(gt))
    #         else:
    #             # gt = np.amax(array_dict[key], axis=1)
    #             # print(np.amax(array_dict[key], axis=1))
    #             myresult = array_dict[key][np.arange(num_images), best_idxs, :][:, None, :]
    #             # print(myresult.shape)

    all_stats = []
    for fname in fnames:
        array_dict = np.load(os.path.join(args.path, fname))[()]
        # print(fname)
        all_stats.append(calculate_stats(array_dict, fname))

    write_to_csv(args.fname, all_stats)
