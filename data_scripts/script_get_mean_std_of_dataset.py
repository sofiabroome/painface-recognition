import sys
sys.path.append('..')
import os
import subprocess
import numpy as np
from tqdm import tqdm
from matplotlib.pyplot import imread

NB_CHANNELS = 3

rgb_folders = ['../data/pf/jpg_128_128_2fps/', '../data/lps/jpg_128_128_2fps/']
flow_folders = ['../data/pf/jpg_128_128_16fps_OF_magnitude_cv2/', '../data/lps/jpg_128_128_16fps_OF_magnitude_cv2_2fpsrate/']

all_folders = rgb_folders + flow_folders
out_folder = '../data/dataset_statistics/'
skip_num = 1

if not os.path.exists(out_folder):
    subprocess.call(['mkdir', out_folder])


def get_all_jpg_paths(frame_folder):
    paths = []
    for root, dirs, files in os.walk(frame_folder):
        for name in files:
            if name.endswith((".jpg")):
                path = os.path.join(root, name)
                paths.append(path)
    return paths

path_lists = []
len_datasets = []
for rgbf in rgb_folders:
    path_list = get_all_jpg_paths(rgbf)
    path_list = path_list[::skip_num]
    len_datasets.append(len(path_list))
    path_lists.append(path_list)

for flowf in flow_folders:
    path_list = get_all_jpg_paths(flowf)
    path_list = path_list[::skip_num]
    len_datasets.append(len(path_list))
    path_lists.append(path_list)


# Read images
means = []
stds = []
for i in range(len(path_lists)):
    mean = np.zeros(NB_CHANNELS)
    std = np.zeros(NB_CHANNELS)
    pl = path_lists[i]

    with tqdm(total=len(pl)) as pbar:
        for idx, path in enumerate(pl):
            pbar.update(1)
            img = imread(path)
            ar = np.asarray(img)/255
            for channel in range(NB_CHANNELS):
                mean[channel] += np.mean(ar[:, :, channel])
                std[channel] += np.std(ar[:, :, channel])
    mean = np.round(mean/len_datasets[i], 3)
    std = np.round(std/len_datasets[i], 3)
    means.append(mean)
    stds.append(std)

# for i in range(len(path_lists)):

nb_rgb = len_datasets[0] + len_datasets[1]
nb_flow = len_datasets[2] + len_datasets[3]

# means [array([m1,m2,m3])x4]
# stds [array([s1,s2,s3])x4]
# len_datasets [len x4]

both_datasets_rgb_mean = (len_datasets[0] * means[0] + len_datasets[1] * means[1])/nb_rgb
both_datasets_rgb_std = (len_datasets[0] * stds[0] + len_datasets[1] * stds[1])/nb_rgb

both_datasets_flow_mean = (len_datasets[2] * means[2] + len_datasets[3] * means[3])/nb_flow
both_datasets_flow_std = (len_datasets[2] * stds[2] + len_datasets[3] * stds[3])/nb_flow


def get_weighted_average(list_of_values, weights):
    wavg = [np.round(a * b, 3) for a, b in zip(list_of_values, weights)]
    return sum(wavg)

wweights = [ww / (nb_rgb + nb_flow) for ww in len_datasets]
avg_means = get_weighted_average(means, wweights)
avg_stds = get_weighted_average(stds, wweights)


f = open(os.path.join(out_folder, 'stats_skip{}.txt'.format(skip_num)), 'w+')
for ind, mm in enumerate(means):
    out_str = 'dataset index: {}, mean: {}, std: {}\n'.format(ind, mm, stds[ind])
    # out_str = 'dataset index: {ind}, mean: {mm}, std: {0:.3f}\n'.format(ind, mm, stds[ind])
    f.write(out_str)

out_str = 'both rgb, mean: {}, std: {}\n'.format(both_datasets_rgb_mean, both_datasets_rgb_std)
f.write(out_str)
out_str = 'both flow, mean: {}, std: {}\n'.format(both_datasets_flow_mean, both_datasets_flow_std)
f.write(out_str)
out_str = 'both datasets, rgb and flow mixed, mean: {}, std: {}\n'.format(avg_means, avg_stds)
f.write(out_str)
