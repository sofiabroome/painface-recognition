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
        # y_pred = model.predict_classes(X_test, batch_size=model.batch_size)
        y_pred = model.predict_generator(test_generator,
                                         steps=int(nb_test_samples/self.batch_size)-1,
                                         verbose=1)
        #y_pred = model.predict_generator(test_generator,
        #                                 steps=10,
        #                                 verbose=1)
        if self.acc:
            scores = model.evaluate_generator(eval_generator,
                                              steps=int(nb_test_samples/self.batch_size)-1)
        # scores = model.evaluate_generator(eval_generator,
        #                          steps=10)
        # y_pred = model.predict_generator(test_generator, steps=3)
        return y_pred, scores

    def evaluate(self, model, y_test, y_pred, scores, args):
        print('Accuracy: ', scores[1])
        file_identifier = args.image_identifier
        y_test = np_utils.to_categorical(y_test, num_classes=args.nb_labels)
        y_preds = np.argmax(y_pred, axis=1)
        y_pred = np_utils.to_categorical(y_preds, num_classes=args.nb_labels)
        nb_preds = len(y_pred)
        y_test = y_test[:nb_preds]
        if self.cr:
            cr = classification_report(y_test, y_pred)
            f = open(_make_cr_filename(args, file_identifier), 'w')
            print(cr, end="", file=f)
            f.close()
            print(cr)

        if self.cm:
            cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
            print(cm)
            f = open(_make_cm_filename(args, file_identifier), 'w')
            print(cm, end="", file=f)
            f.close()

    def classification_report(self, y_test, y_pred):
        return classification_report(np.argmax(y_test, axis=1), y_pred,
                                     target_names=self.target_names,
                                     digits=NB_DECIMALS)

    def confusion_matrix(self, y_test, y_pred):
        return confusion_matrix(np.argmax(y_test, axis=1), y_pred)


def _make_cr_filename(args, identifier):
    return args.model + "_" + identifier + "_NB_LSTM_UNITS_" +\
                  str(args.nb_lstm_units) + "_NB_CONV_FILTERS_" +\
                  str(args.nb_conv_filters) + "_CLASSREPORT.txt"


def _make_cm_filename(args, identifier):
    return args.model + "_" + identifier + "_NB_LSTM_UNITS_" + \
                  str(args.nb_lstm_units) + "_NB_CONV_FILTERS_" + \
                  str(args.nb_conv_filters) + "_CONFMAT.txt"
