import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

overview_df = pd.read_csv('../metadata/lps_videos_overview.csv')
resampled_df = pd.read_csv('../data/lps/video_level_stats/summary_resample.csv')
noresampled_df = pd.read_csv('../data/lps/video_level_stats/summary_noresample.csv')


merged_resample = pd.merge(overview_df, resampled_df, left_on='video_id', right_on='video_id')
merged_noresample = pd.merge(overview_df, noresampled_df, left_on='video_id', right_on='video_id')

pain_merged_noresample = merged_noresample.loc[merged_noresample['pain'] > 0].reset_index()
nopain_merged_noresample = merged_noresample.loc[merged_noresample['pain'] == 0].reset_index()
pain_merged_noresample.rename(columns={'length_y': 'pain_noresample_lengths'}, inplace=True)
nopain_merged_noresample.rename(columns={'length_y': 'nopain_noresample_lengths'}, inplace=True)

pain_merged_resample = merged_resample.loc[merged_resample['pain'] > 0].reset_index()
nopain_merged_resample = merged_resample.loc[merged_resample['pain'] == 0].reset_index()
pain_merged_resample.rename(columns={'length_y': 'pain_resample_lengths'}, inplace=True)
nopain_merged_resample.rename(columns={'length_y': 'nopain_resample_lengths'}, inplace=True)

bins = list(range(0, 150, 5))
fig, ax = plt.subplots()
np_noresample_heights, a_bins = np.histogram(nopain_merged_noresample['nopain_noresample_lengths'], bins=bins)
p_noresample_heights, b_bins = np.histogram(pain_merged_noresample['pain_noresample_lengths'], bins=a_bins)
width = (a_bins[1] - a_bins[0])/3

import ipdb; ipdb.set_trace()

npnr = nopain_merged_noresample['nopain_noresample_lengths'].values
pnr = pain_merged_noresample['pain_noresample_lengths'].values

npr = nopain_merged_resample['nopain_resample_lengths'].values
pr = pain_merged_resample['pain_resample_lengths'].values

fig, ax = plt.subplots(1,2,figsize=(20,5))
alpha=0.6
ax[0].hist(npnr, a_bins,alpha=alpha,label='No pain, no resample')
ax[0].hist(pnr, b_bins,alpha=alpha,label='Pain, no resample')
ax[0].set_title('Number of clip-features per video', fontsize=15)
ax[1].hist(npr, a_bins,alpha=alpha,label='No pain, resample')
ax[1].hist(pr, b_bins,alpha=alpha,label='Pain, resample')
ax[1].set_title('Number of clip-features per video', fontsize=15)
# ax[2].hist(np_noresample_heights, a_bins,alpha=alpha,label='No pain, no resample')
# ax[2].hist(p_noresample_heights, b_bins,alpha=alpha,label='Pain, no resample')
# ax[2].set_title('Video lengths', fontsize=15)
for i in range(2):
    ax[i].legend(prop={'size':'x-large'})
plt.subplots_adjust(wspace=0.15, hspace=0)
plt.show()


