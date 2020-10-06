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
    mean = np.zeros((NB_CHANNELS))
    std = np.zeros((NB_CHANNELS))
    pl = path_lists[i]

    with tqdm(total=len(pl)) as pbar:
        for idx, path in enumerate(pl):
            pbar.update(1)
            img = imread(path)
            ar = np.asarray(img)/255
            # import ipdb; ipdb.set_trace()
            # ar /= 255.0
            for channel in range(NB_CHANNELS):
                mean[channel] += np.mean(ar[:, :, channel])
                std[channel] += np.std(ar[:, :, channel])
    mean = np.round(mean/len_datasets[i], 3)
    std = np.round(std/len_datasets[i], 3)
    means.append(mean)
    stds.append(std)
    

f = open(os.path.join(out_folder, 'stats_skip{}.txt'.format(skip_num)), 'w+')
for ind, mm in enumerate(means):
    out_str = 'dataset index: {}, mean: {}, std: {}\n'.format(ind, mm, stds[ind])
    # out_str = 'dataset index: {ind}, mean: {mm}, std: {0:.3f}\n'.format(ind, mm, stds[ind])
    f.write(out_str)
    
