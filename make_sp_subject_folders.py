import pandas as pd
import subprocess


def make_folders():
    """
    This method only needs to be run once. It creates the per-horse
    and per-sequence folders where to save the computed optical flow.
    :return: None
    """

    for subject_id in subject_ids:
        print("NEW SUBJECT")
        subj_dir_path = dir_in_which + subject_id
        subprocess.call(['mkdir', subj_dir_path])


if __name__ == '__main__':
    dir_in_which = 'data/ShoulderPain172x129_OF_cv2/'
    subject_ID_df = pd.read_csv('shoulder_pain_subjects.csv')
    subject_ids = subject_ID_df['Subject'].values
    
    make_folders()

