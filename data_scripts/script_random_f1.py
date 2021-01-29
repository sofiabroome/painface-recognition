import numpy as np

def get_one_random_row(nb_samples, best_case, pain):
    """ nb_samples: int
        best_case: bool
        pain: bool
    """
    if nb_samples % 2 == 0:
        row = [nb_samples/2, nb_samples/2]
    else:
        if pain:
            row = [int(nb_samples/2), int(nb_samples/2) + 1]
            if not best_case:
                row = [row[1], row[0]]
        else:
            row = [int(nb_samples/2) + 1, int(nb_samples/2)]
            if not best_case:
                row = [row[1], row[0]]
    return row

def construct_random_confusion_matrix(nb_pain, nb_nopain, best_case):
    """ args: 
        nb_pain, nb_nopain: int

        return: list of lists.
        [[true no pain, false no pain],
         [false pain, true pain]]"""

    row_nopain = get_one_random_row(nb_nopain, best_case=best_case, pain=False)
    row_pain = get_one_random_row(nb_pain, best_case=best_case, pain=True)

    cm = [row_nopain, row_pain]
    
    return cm

def get_f1(precision, recall):
    return 2*precision*recall/(precision+recall)

def get_macro_avg_f1_from_confmat(cm):
    nopain_recall = cm[0][0]/nb_nopain
    nopain_precision = cm[0][0]/(cm[0][0] + cm[1][0])
    nopain_f1 = get_f1(nopain_precision, nopain_recall)

    pain_recall = cm[1][1]/nb_pain
    pain_precision = cm[1][1]/(cm[0][1] + cm[1][1])
    pain_f1 = get_f1(pain_precision, pain_recall)

    macro_avg_f1 = (nopain_f1 + pain_f1)/2

    return macro_avg_f1

nb_pain = int(input('Nb pain: '))
nb_nopain = int(input('Nb nopain: '))

best_case_cm = construct_random_confusion_matrix(nb_pain, nb_nopain, best_case=True)
print('Best case cm: ', best_case_cm)
worst_case_cm = construct_random_confusion_matrix(nb_pain, nb_nopain, best_case=False)
print('Worst case cm: ', worst_case_cm)

best_case_f1 = get_macro_avg_f1_from_confmat(best_case_cm)
worst_case_f1 = get_macro_avg_f1_from_confmat(worst_case_cm)

print('Best case macro avg F1: ', round(best_case_f1, 4))
print('Worst case macro avg F1: ', round(worst_case_f1, 4))
print('Average of the two: ', round((best_case_f1 + worst_case_f1)/2, 4))
