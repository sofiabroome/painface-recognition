from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import subprocess
import wandb
import os

NB_DECIMALS = 4


class Evaluator:
    def __init__(self, acc, cm, cr, auc, target_names, batch_size):
        self.acc = acc
        self.cr = cr
        self.cm = cm
        self.auc = auc
        self.target_names = target_names
        self.batch_size = batch_size
        self.test_set = None

    def set_test_set(self, df_test):
        """
        :param df_test: pd.DataFrame with all info about the test set.
        """
        self.test_set = df_test

    def test(self, model, test_generator, eval_generator, nb_steps, X_test=None):
        ###### If not a generator:
        #    y_pred = model.predict_classes(X_test, batch_size=model.batch_size)

        y_pred = model.predict(x=test_generator,
                               steps=nb_steps,
                               verbose=1)
        if self.acc:
            scores = model.evaluate(x=eval_generator,
                                    steps=nb_steps,
                                    return_dict=True)
        return y_pred, scores

    def evaluate(self, model, y_test, y_pred, softmax_predictions,
                 scores, config_dict, y_paths):
        """
        Compute confusion matrix and class report with F1-scores.
        :param model: Model object
        :param y_test: np.ndarray (dim, seq_length, nb_classes)
        :param y_pred: np.ndarray (dim, seq_length, nb_classes)
        :param scores: [np.ndarray]
        :param config_dict: dict
        :param y_paths: [str]
        :return: None
        """
        print('Scores: ', scores)
        print('Model metrics: ', model.metrics_names)
        assert(y_test.shape == y_pred.shape)

        if len(y_pred.shape) > 2: # If sequential data
            y_pred, paths = get_majority_vote_3d(y_pred, y_paths)
            softmax_predictions, _ = get_majority_vote_3d(softmax_predictions, y_paths)
            y_test, _ = get_majority_vote_3d(y_test, y_paths)
        # self.look_at_classifications(y_test, y_pred, paths, softmax_predictions)
        nb_preds = len(y_pred)
        nb_tests = len(y_test)

        if nb_preds != nb_tests:
            print("Warning, number of predictions not the same as the length of the y_test vector.")
            print("Y test length: ", nb_tests)
            print("Y pred length: ", nb_preds)
            if nb_preds < nb_tests:
                y_test = y_test[:nb_preds]
            else:
                y_pred = y_pred[:nb_tests]

        # Print labels and predictions.
        print('y_test.shape: ', y_test.shape)
        print('y_pred.shape: ', y_pred.shape)
        # print('y_test and y_test.shape: ', y_test, y_test.shape)
        # print('y_pred and y_pred.shape: ', y_pred, y_pred.shape)

        self.print_and_save_evaluations(y_test, y_pred, softmax_predictions, config_dict)

    def look_at_classifications(self, y_true, y_pred, paths, softmax_predictions):
        TP_ind = get_index_of_type_of_classification(y_true, y_pred, true=1, pred=1)
        confidence_level = softmax_predictions[TP_ind]
        print('Sequence starting with ', paths[TP_ind], 'was a true positive with confidence ', confidence_level)

        FP_ind = get_index_of_type_of_classification(y_true, y_pred, true=0, pred=1)
        confidence_level = softmax_predictions[FP_ind]
        print('Sequence starting with ', paths[FP_ind], 'was a false positive with confidence ', confidence_level)

        TN_ind = get_index_of_type_of_classification(y_true, y_pred, true=0, pred=0)
        confidence_level = softmax_predictions[TN_ind]
        print('Sequence starting with ', paths[TN_ind], 'was a true negative with confidence ', confidence_level)

        FN_ind = get_index_of_type_of_classification(y_true, y_pred, true=1, pred=0)
        confidence_level = softmax_predictions[FN_ind]
        print('Sequence starting with ', paths[FN_ind], 'was a false negative with confidence ', confidence_level)

    def print_and_save_evaluations(self, y_test, y_pred, softmax_predictions, config_dict):
        """
        :params y_pred, y_test: 2D-arrays [nsamples, nclasses]
        """
        if self.cr:
            cr = classification_report(y_test, y_pred,
                                       target_names=self.target_names,
                                       digits=NB_DECIMALS)
            f = open(self._make_filename('classreport', config_dict), 'w')
            print(cr, end="", file=f)
            f.close()
            print(cr)
            cr = classification_report(y_test, y_pred,
                                       target_names=self.target_names,
                                       digits=NB_DECIMALS,
                                       output_dict=True)
            wandb.log({'test f1-score' : cr['macro avg']['f1-score']})

        if self.cm:
            cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
            print(cm)
            correct = np.sum([ar[i] for i, ar in enumerate(cm)])
            total_samples = np.sum(cm)
            acc = round(correct/total_samples, NB_DECIMALS)
            wandb.log({'test accuracy' : acc})
            print(acc, ' acc.')
            f = open(self._make_filename('confmat', config_dict), 'w')
            print(cm,' ', acc, ' acc.', end="", file=f)
            f.close()

        if self.auc:
            auc_weighted = round(
                roc_auc_score(y_test, softmax_predictions, average='weighted'), NB_DECIMALS)
            auc_macro = round(
                roc_auc_score(y_test, softmax_predictions, average='macro'), NB_DECIMALS)
            auc_micro = round(
                roc_auc_score(y_test, softmax_predictions, average='micro'), NB_DECIMALS)
            print('Weighted AUC: ', auc_weighted)
            print('Macro AUC: ', auc_macro)
            print('Micro AUC: ', auc_micro)
            wandb.log({'test AUC weighted' : auc_weighted})
            wandb.log({'test AUC macro' : auc_macro})
            wandb.log({'test AUC micro' : auc_micro})

            with open(config_dict['model'] + "_" + config_dict['job_identifier'] +'_auc' + '.txt', 'w') as f:
                print('Weighted AUC: ', auc_weighted, file=f)
                print('Macro AUC: ', auc_macro, file=f)
                print('Micro AUC: ', auc_micro, file=f)
                f.close()

    def _make_filename(self, thing_to_save, config_dict):
        if not os.path.exists('results/'):
           subprocess.call(['mkdir', 'results']) 
        return 'results/' + thing_to_save + config_dict['model'] + \
               '_' + config_dict['job_identifier'] + '.txt'


def get_index_of_type_of_classification(y_true, y_pred, true=1, pred=1):
    for index, value in enumerate(y_true):
        halfway = int(len(y_true)/2)
        if index > 300:
            if np.argmax(value) == true:
                if np.argmax(y_pred[index]) == pred:
                    return index


def get_majority_vote_for_sequence(sequence, nb_classes):
    """
    Get the most common class for one sequence.
    :param sequence:
    :return:
    """
    votes_per_class = np.zeros((nb_classes, 1))
    for i in range(len(sequence)):
        class_vote = np.argmax(sequence[i])
        votes_per_class[class_vote] += 1
    # Return random choice of the max if there's a tie.
    return np.random.choice(np.flatnonzero(votes_per_class == votes_per_class.max()))


def get_majority_vote_3d(y_pred, y_paths):
    """
    Take the majority vote for every sequence.
    If there's a tie the choice is randomized.
    :param y_pred: Array with 3 dimensions.
    :return: Array with 2 dims.
    """
    nb_samples = y_pred.shape[0]
    nb_classes = y_pred.shape[2]
    majority_votes = np.zeros((nb_samples, nb_classes))
    corresponding_paths = []
    for i in range(nb_samples):
        sample = y_pred[i]
        corresponding_paths.append(y_paths[i])
        class_sums = []
        for c in range(nb_classes):
            # Sum of votes for one class across sequence
            class_sum = sample[:,c].sum()
            class_sums.append(class_sum)
        cs = np.array(class_sums)
        max_class = np.random.choice(np.flatnonzero(cs == cs.max()))
        majority_votes[i, max_class] = 1
    return majority_votes, corresponding_paths

