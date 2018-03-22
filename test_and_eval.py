import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from keras.utils import np_utils

NB_DECIMALS = 4


class Evaluator:
    def __init__(self, acc, cm, cr, auc, target_names, batch_size):
        self.acc = acc
        self.cr = cr
        self.cm = cm
        self.auc = auc
        self.target_names = target_names
        self.batch_size = batch_size

    def test(self, model, args, test_generator, eval_generator, nb_steps, X_test=None):
        ###### If not a generator:
        #    y_pred = model.predict_classes(X_test, batch_size=model.batch_size)

        y_pred = model.predict_generator(test_generator,
                                         steps=nb_steps,
                                         verbose=1)
        if self.acc:
            scores = model.evaluate_generator(eval_generator,
                                              steps=nb_steps)
        return y_pred, scores

    def evaluate(self, model, y_test, y_pred, softmax_predictions, scores, args):
        """
        Compute confusion matrix and class report with F1-scores.
        :param model: Model object
        :param y_test: np.ndarray (dim, seq_length, nb_classes)
        :param y_pred: np.ndarray (dim, seq_length, nb_classes)
        :param scores: [np.ndarray]
        :param args: command line args
        :return: None
        """
        print('Scores: ', scores)
        print('Model metrics: ', model.metrics_names)
        assert(y_test.shape == y_pred.shape)

        if len(y_pred.shape) > 2: # If sequential data
            y_pred = get_majority_vote_3d(y_pred)
            softmax_predictions = get_majority_vote_3d(softmax_predictions)
            y_test = get_majority_vote_3d(y_test)

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

        self.print_and_save_evaluations(y_test, y_pred, softmax_predictions, args)


    def print_and_save_evaluations(self, y_test, y_pred, softmax_predictions, args):
        """
        :params y_pred, y_test: 2D-arrays [nsamples, nclasses]
        """
        if self.cr:
            cr = classification_report(y_test, y_pred,
                                       target_names=self.target_names,
                                       digits=NB_DECIMALS)
            f = open(_make_cr_filename(args), 'w')
            print(cr, end="", file=f)
            f.close()
            print(cr)

        if self.cm:
            if args.nb_labels != 2 or args.nb_input_dims == 5:
                cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
            else:
                cm = confusion_matrix(y_test, y_pred)
            print(cm)
            f = open(_make_cm_filename(args), 'w')
            print(cm, end="", file=f)
            f.close()

        if self.auc:
            if args.nb_input_dims == 4:
                y_test = np.array([np_utils.to_categorical(x,num_classes=args.nb_labels) for x in y_test])
                y_test = np.reshape(y_test, (-1, args.nb_labels))
            auc_weighted = roc_auc_score(y_test, softmax_predictions, average='weighted')
            auc_macro = roc_auc_score(y_test, softmax_predictions, average='macro')
            auc_micro = roc_auc_score(y_test, softmax_predictions, average='micro')
            print('Weighted AUC: ', auc_weighted)
            print('Macro AUC: ', auc_macro)
            print('Micro AUC: ', auc_micro)

            with open('auc' + args.image_identifier +'.txt', 'w') as f:
                # print('Filename:', filename, file=f) 
                print('Weighted AUC: ', auc_weighted, file=f)
                print('Macro AUC: ', auc_macro, file=f)
                print('Micro AUC: ', auc_micro, file=f)
                f.close()


def _make_cr_filename(args):
    return args.model + "_" + args.image_identifier + "_LSTM_UNITS_" +\
                  str(args.nb_lstm_units) + "_CONV_FILTERS_" +\
                  str(args.nb_conv_filters) + "_CR.txt"


def _make_cm_filename(args):
    return args.model + "_" + args.image_identifier + "_LSTM_UNITS_" + \
                  str(args.nb_lstm_units) + "_CONV_FILTERS_" + \
                  str(args.nb_conv_filters) + "_CM.txt"


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


def get_majority_vote_3d(y_pred):
    """
    I want to take the majority vote for every sequence.
    If there's a tie the choice is randomized.
    :param y_pred: Array with 3 dimensions.
    :return: Array with 2 dims.
    """
    nb_samples = y_pred.shape[0]
    nb_classes = y_pred.shape[2]
    majority_votes = np.zeros((nb_samples, nb_classes))
    for i in range(nb_samples):
        sample = y_pred[i]
        class_sums = []
        for c in range(nb_classes):
            # Sum of votes for one class across sequence
            class_sum = sample[:,c].sum()
            class_sums.append(class_sum)
        cs = np.array(class_sums)
        max_class = np.random.choice(np.flatnonzero(cs == cs.max()))
        majority_votes[i, max_class] = 1
    return majority_votes

