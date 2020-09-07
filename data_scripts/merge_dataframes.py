import pandas as pd

df_pain = pd.read_csv('../metadata/lps_videos_overview_old_w_pain.csv')
df_new = pd.read_csv('lps_videos_overview.csv')

import ipdb; ipdb.set_trace()

df_merged = pd.merge(df_pain, df_new, on='video_id', how='right')

df_merged.to_csv('../metadata/lps_videos_overview.csv')
