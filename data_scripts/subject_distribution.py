import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--clip-list-path', nargs='?', type=str,
                    help='Path to csv file listing the clips to summarize.')

args = parser.parse_args()

df = pd.read_csv(args.clip_list_path)

subjects = set(df['subject'].values)

total_pain_time = pd.to_timedelta(0, 's')
total_no_pain_time = pd.to_timedelta(0, 's')

subject_pain_distribution = {} # {str: list of length 2}
for subject in subjects:
    horse_df = df.loc[df['subject'] == subject]
    pain_df = horse_df.loc[horse_df['pain'] > 0]
    pain_seconds = pain_df['length'].sum()
    no_pain_df = horse_df.loc[horse_df['pain'] == 0]
    no_pain_seconds = no_pain_df['length'].sum()

    no_pain_time = pd.to_timedelta(no_pain_seconds, 's')
    pain_time = pd.to_timedelta(pain_seconds, 's')

    total_pain_time +=  pain_time
    total_no_pain_time +=  no_pain_time

    subject_pain_distribution[subject] = [no_pain_time, pain_time]

for key, value in subject_pain_distribution.items():
    print(key, value, '\n')

print('Total no pain time: ', total_no_pain_time)
print('Total pain time: ', total_pain_time)
    
