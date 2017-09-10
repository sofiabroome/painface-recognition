import numpy as np
import sklearn

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

    def test(self, model, test_generator, eval_generator, nb_test_samples, X_test=None):
        ## y_pred = model.predict_classes(X_test, batch_size=model.batch_size)
        y_pred = model.predict_generator(test_generator,
                                         steps=int(nb_test_samples/self.batch_size),
                                         verbose=1)
        # y_pred = model.predict_generator(test_generator,
        #                                  steps=10,
        #                                  verbose=1)
        if self.acc:
            scores = model.evaluate_generator(eval_generator,
                                              steps=int(nb_test_samples/self.batch_size))
            # scores = model.evaluate_generator(eval_generator,
            #                                   steps=10)
        return y_pred, scores

    def evaluate(self, model, y_test, y_pred, scores, args):
        print('Accuracy: ', scores[1])
        file_identifier = args.image_identifier
        if args.nb_labels != 2:
            y_test = np_utils.to_categorical(y_test, num_classes=args.nb_labels)
        y_pred = np.argmax(y_pred, axis=1)
        if args.nb_labels != 2:
            y_pred = np_utils.to_categorical(y_pred, num_classes=args.nb_labels)
        nb_preds = len(y_pred)
        nb_tests = len(y_test)
        if nb_preds != nb_tests:
            print("Warning, number of predictions not the same as the length of the y_test vector.")
            print("Y test length: ", nb_tests)
            print("Y pred length: ", nb_preds)
        y_test = y_test[:nb_preds]
        print('y_test:')
        print(y_test)
        print('y_pred:')
        print(y_pred)
        if self.cr:
            cr = classification_report(y_test, y_pred)
            f = open(_make_cr_filename(args, file_identifier), 'w')
            print(cr, end="", file=f)
            f.close()
            print(cr)

        if self.cm:
            if args.nb_labels != 2:
                cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
            else:
                cm = confusion_matrix(y_test, y_pred)
            print(cm)
            f = open(_make_cm_filename(args, file_identifier), 'w')
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


def _make_cr_filename(args, identifier):
    return args.model + "_" + identifier + "_NB_LSTM_UNITS_" +\
                  str(args.nb_lstm_units) + "_NB_CONV_FILTERS_" +\
                  str(args.nb_conv_filters) + "_CLASSREPORT.txt"


def _make_cm_filename(args, identifier):
    return args.model + "_" + identifier + "_NB_LSTM_UNITS_" + \
                  str(args.nb_lstm_units) + "_NB_CONV_FILTERS_" + \
                  str(args.nb_conv_filters) + "_CONFMAT.txt"
