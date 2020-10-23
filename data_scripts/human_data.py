import pandas as pd
import numpy as np

path_to_cleaner_data = '~/Downloads/vets_results_cleaned_up.csv'


def prepare_raw_data(path_to_raw_csv):
    df = pd.read_csv(path_to_raw_csv)

    # Part 1: Drop irrelevant columns and rename columns to shorter names
    # i.e. not the full question from Canvas
    df.drop('section')
    df.drop(columns=['section'])
    df = df.drop(columns=['section'])
    df = df.drop(columns=['section_id'])
    df = df.drop(columns=['attempt'])
    df.rename(columns={
        '69192: Vänligen fyll i den individuella sifferkoden som du fått'
        'på mail tillsammans med länken till det här quizet:':
            'sifferkod'}, inplace=True)
    df.rename(columns={'65723: Kön:': 'gender'}, inplace=True)
    # Cols named 'x.y.z' were the scores of the questions = drop, not relevant.
    df = df.drop(columns=['0.0'])
    df = df.drop(columns=['0.0.1'])
    df = df.drop(columns=['0.0.2'])
    df = df.drop(columns=['0.0.3'])
    df.rename(columns={'65724: Antal år verksam som veterinär:': 'years_active_as_vet'}, inplace=True)
    df.rename(columns={'65725: Använder du någon form av smärtskala i ditt arbete?': 'use_painscale_at_work'},
              inplace=True)
    df.rename(columns={
        '68929: Notera: Den här frågan är bara till för övning. Efter denna'
        'fråga kommer de 25 videoklippen du ska smärtbedöma.\n \nPå en skala'
        'från 0-10, hur smärtpåverkad bedömer du att hästen på filmen är?\nEndast'
        '0 betyder "ingen smärta".\ntest_clip_0.mp4': 'practice_question'},
              inplace=True)
    df = df.drop(columns=['0.0.4'])
    df = df.drop(columns=['1.0.1'])
    df.to_csv('intermediate_save.csv')
    df = df.drop(columns=['1.0.3'])
    df = df.drop(columns=['1.0.4'])
    df = df.drop(columns=['1.0.5'])
    df = df.drop(columns=['1.0.6'])
    df = df.drop(columns=['1.0.7'])
    df = df.drop(columns=['1.0.8'])
    df = df.drop(columns=['1.0.9'])
    df = df.drop(columns=['1.0.10'])
    df = df.drop(columns=['1.0.11'])
    df = df.drop(columns=['1.0.12'])
    df = df.drop(columns=['1.0.13'])
    df = df.drop(columns=['1.0.14'])
    df = df.drop(columns=['1.0.15'])
    df = df.drop(columns=['1.0.16'])
    df = df.drop(columns=['1.0.17'])
    df = df.drop(columns=['1.0.18'])
    df = df.drop(columns=['1.0.19'])
    df = df.drop(columns=['1.0.20'])
    df = df.drop(columns=['1.0.21'])
    df = df.drop(columns=['1.0.22'])
    df = df.drop(columns=['1.0.23'])
    df = df.drop(columns=['1.0.24'])
    df = df.drop(columns=['1.0'])
    df = df.drop(columns=['1.0.2'])
    df = df.drop(columns=['n correct'])
    df = df.drop(columns=['n incorrect'])
    df = df.drop(columns=['score'])
    cols = list(df.columns)
    qstrings = ['q' + str(i) for i in range(1, 26)]
    new_cols = ['submitted', 'sifferkod', 'gender', 'years_active_as_vet', 'use_painscale_at_work',
                'practice_question'] + qstrings
    df.columns = new_cols

    # Part 2, insert a ground truth entry
    gtdf = pd.read_csv[
        '../Documents/EquineML/painface-recognition/data/lps/random_clips_lps/ground_truth_randomclips_lps.csv']
    gtdf.sort_values(by='ind', axis=1)
    gtdf = pd.read_csv(
        '../Documents/EquineML/painface-recognition/data/lps/random_clips_lps/ground_truth_randomclips_lps.csv')
    gtdf.sort_values(by=['ind'], axis=1)
    gtdf.sort_values(by=['ind'])
    gt = gtdf.sort_values(by=['ind'])
    pain_list = list(gt['pain'])
    gt_list = ['nan', 'gt', 'gender', '0', 'no', 'no']
    gt_entry = gt_list + pain_list
    df.append(gt_entry)
    dflength = len(df)
    df.loc[dflength] = gt_entry
    gt_copy = gt
    gt_copy.drop(15)
    gt_copy.drop(28)
    df.drop(28)
    df_without_me = df.drop(28)
    df = df_without_me
    df.to_csv('qrename_intermediate_save.csv')

    df.to_csv(path_to_cleaner_data)


def compute_results():
    df = pd.read_csv(path_to_cleaner_data)

    q_col_names = ['q' + str(i) for i in range(1, 26)]

    NB_PARTICIPANTS = 28
    GT_ROW = 28

    OUTLIER = 467

    acceptance_threshold = 0

    scores = []
    pain_scores = []
    nopain_scores = []
    for participant in range(0, NB_PARTICIPANTS):
        nb_correct = 0
        nb_correct_pain = 0
        nb_correct_nopain = 0
        code_id = df['sifferkod'][participant]
        if int(code_id) == OUTLIER:
            continue

        for q in q_col_names:
            q_answers = df[q]
            ground_truth = q_answers[GT_ROW]
            answer = q_answers[participant]

            pain = 1 if ground_truth > 0 else 0
            print('Label: ', ground_truth, 'Pain: ', pain)

            if pain:
                if answer > acceptance_threshold:
                    correct = 1
                    pain_correct = 1
                else:
                    correct = 0
                    pain_correct = 0
                nb_correct_pain += pain_correct
            if not pain:
                if answer > acceptance_threshold:
                    correct = 0
                    nopain_correct = 0
                else:
                    correct = 1
                    nopain_correct = 1
                nb_correct_nopain += nopain_correct

            print('Answer:', answer, 'Correct: ', correct, '\n')
            nb_correct += correct
        scores.append(nb_correct)
        pain_scores.append(nb_correct_pain)
        nopain_scores.append(nb_correct_nopain)

        print('\n Participant {} was correct {} times'.format(participant, nb_correct))
        print('{} for pain and {} for no pain'.format(nb_correct_pain, nb_correct_nopain))

    print('All scores: ', np.array(scores)/25)
    print('Best participant: ', np.argmax(np.array(scores)))
    print('Mean score: ', round(np.mean(np.array(scores)/25), 4))
    print('Std score: ', round(np.std(np.array(scores)/25), 4))

    print('\nMean pain score: ', round(np.mean(np.array(pain_scores)/13), 4))
    print('Std pain score: ', round(np.std(np.array(pain_scores)/13), 4))

    print('\nMean no pain score: ', round(np.mean(np.array(nopain_scores)/12), 4))
    print('Std no pain score: ', round(np.std(np.array(nopain_scores)/12), 4))

