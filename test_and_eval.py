import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import np_utils

NB_DECIMALS = 4


class Evaluator:

    def __init__(self, acc, cm, cr, target_names, batch_size):
        self.acc = acc
        self.cr = cr
        self.cm = cm
        self.target_names = target_names
        self.batch_size = batch_size

    def test(self, model, args, test_generator, eval_generator, nb_test_samples, X_test=None):
        ###### If not a generator:
        #    y_pred = model.predict_classes(X_test, batch_size=model.batch_size)
        ws = args.seq_length  # "Window size" in a sliding window.
        ss = args.seq_stride  # Provide argument for slinding w. stride.

        # SET TEST STEPS
        if args.nb_input_dims == 5:
            if args.seq_stride == args.seq_length:
                nb_steps = int(nb_test_samples / (args.batch_size * args.seq_length))
            else:
                valid_test = nb_test_samples - (ws - 1)
                nw_test = valid_test // ss  # Number of windows
                nb_steps = int(nw_test / args.batch_size)
        if args.nb_input_dims == 4:
            nb_steps = int(nb_test_samples/args.batch_size)
        if args.test_run == 1:
            nb_steps = 2

        y_pred = model.predict_generator(test_generator,
                                             steps=nb_steps,
                                             verbose=1)
        if self.acc:
            scores = model.evaluate_generator(eval_generator,
                                                  steps=nb_steps)
        return y_pred, scores

    def evaluate(self, model, y_test, y_pred, scores, args):
        """
        Compute confusion matrix and class report with F1-scores.
        :param model: Model object
        :param y_test: np.ndarray (dim,)
        :param y_pred: np.ndarray (dim, nb_classes)
        :param scores: [np.ndarray]
        :param args: command line args
        :return: None
        """
        print('Scores: ', scores)
        print('Model metrics: ', model.metrics_names)

        if len(y_pred.shape) > 2: # If sequential data
            y_pred = np.argmax(y_pred, axis=2)
            y_pred_list = [np_utils.to_categorical(x, num_classes=args.nb_labels) for x in y_pred]
            y_pred = np.array(y_pred_list)
            y_pred = get_majority_vote_3d(y_pred)
        else:                     # If still frames
            y_pred = np.argmax(y_pred, axis=1)

        if args.nb_labels != 2 or args.nb_input_dims == 5:
            # If sequences, get majority labels per window.
            y_test = np_utils.to_categorical(y_test, num_classes=args.nb_labels)
            y_test = get_sequence_majority_labels(y_test, args.seq_length, args.seq_stride)

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

        y_test = np_utils.to_categorical(y_test, args.nb_labels)

        # Print labels and predictions.
        print('y_test:', y_test)
        print('y_pred:', y_pred)

        self.print_and_save_evaluations(y_pred, y_test, args)


    def print_and_save_evaluations(self, y_pred, y_test, args):
        """
        :params y_pred, y_test: 2D-arrays [nsamples, nclasses]
        """
        if self.cr:
            cr = classification_report(y_test, y_pred)
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

    def classification_report(self, y_test, y_pred):
        return classification_report(np.argmax(y_test, axis=1), y_pred,
                                     target_names=self.target_names,
                                     digits=NB_DECIMALS)

    def confusion_matrix(self, y_test, y_pred):
        # y_test = np_utils.to_categorical(y_test, num_classes=2)
        # y_pred = np_utils.to_categorical(y_pred, num_classes=2)
        # just the below return line was before 5/9.
        # return confusion_matrix(np.argmax(y_test, axis=1), y_pred)
        return confusion_matrix(y_test, y_pred)




def _make_cr_filename(args):
    return args.model + "_" + args.image_identifier + "_LSTM_UNITS_" +\
                  str(args.nb_lstm_units) + "_CONV_FILTERS_" +\
                  str(args.nb_conv_filters) + "_CR.txt"


def _make_cm_filename(args):
    return args.model + "_" + args.image_identifier + "_LSTM_UNITS_" + \
                  str(args.nb_lstm_units) + "_CONV_FILTERS_" + \
                  str(args.nb_conv_filters) + "_CM.txt"


def get_sequence_majority_labels(y_per_frame, ws, stride):
    """
    Get the majority labels for every sequence.
    :param y_per_frame: np.ndarray
    :param ws: int
    :param stride: int
    :return: np.ndarray
    """

    nb_frames = len(y_per_frame)
    valid = nb_frames - (ws - 1)
    nw = valid // stride

    window_votes = np.zeros((nw, 1))

    for window_index in range(nw):
        start = window_index * stride
        stop = start + ws
        window = y_per_frame[start:stop]
        window_votes[window_index] = get_majority_vote_for_sequence(window,
                                                                    y_per_frame.shape[1])


    return window_votes


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

    return np.argmax(votes_per_class)


def get_majority_vote_3d(y_pred):
    """
    I want to take the majority vote for every sequence.
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
        majority_votes[i, np.argmax(class_sums)] = 1
    return majority_votes
