from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
import numpy as np
import data_handler
import subprocess
import random
import train
import wandb
from tqdm import tqdm
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

    def evaluate(self, model, y_test, y_pred, softmax_predictions,
                 config_dict, y_paths):
        """
        Compute confusion matrix and class report with F1-scores.
        :param model: Model object
        :param y_test: np.ndarray (dim, seq_length, nb_classes)
        :param y_pred: np.ndarray (dim, seq_length, nb_classes)
        :param config_dict: dict
        :param y_paths: [str]
        :return: None
        """
        print('Model metrics: ', model.metrics_names)
        assert(y_test.shape == y_pred.shape)

        if len(y_pred.shape) > 2:  # If sequential data
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
            wandb.log({'Test AUC weighted': auc_weighted})
            wandb.log({'Test AUC macro': auc_macro})
            wandb.log({'Test AUC micro': auc_micro})

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
    :param nb_classes: int
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


def get_per_video_vote(y_pred, y_paths, ground_truth, confidence_threshold=0):
    nb_sequences = y_pred.shape[0]
    majority_votes = {}
    for i in range(nb_sequences):
        seq_path = y_paths[i]
        video_id = data_handler.get_video_id_from_frame_path(seq_path)
        label_votes = majority_votes.get(video_id, {0: 0, 1: 0})
        confidence = np.max(y_pred[i])
        if 'ground_truth' not in label_votes:
            label_votes['ground_truth'] = np.argmax(ground_truth[i])
        if confidence > confidence_threshold:
            vote = np.argmax(y_pred[i])
            if vote in label_votes:
                label_votes[vote] += 1
            else:
                label_votes[vote] = 1
    
            majority_votes[video_id] = label_votes
    print(majority_votes)
    wandb.log({'video votes, confthresh {}'.format(confidence_threshold): str(majority_votes)})
    return majority_votes


def compute_video_level_accuracy(majvotes):
    nb_correct = 0
    total = 0
    nb_correct_mil = {0.05: 0, 0.1: 0, 0.2: 0, 0.3: 0, 0.4: 0}
    mil_thresholds = nb_correct_mil.keys()
    for video_id in majvotes.keys():
        nopain = majvotes[video_id][0]
        pain = majvotes[video_id][1]
        nb_instances = pain + nopain
        majority = 0 if nopain > pain else 1
        if nopain == pain:
            majority = random.choice([0, 1])
        gt = majvotes[video_id]['ground_truth']
        for mil_threshold in mil_thresholds:
            if gt == 1:
                correct = 1 if pain >= int(mil_threshold*nb_instances) else 0
            else:
                correct = 1 if pain < int(mil_threshold*nb_instances) else 0
            # if correct:
                # print('{} was correctly classified by MIL-vote, threshold {}'.format(
                # video_id, mil_threshold))
                
            nb_correct_mil[mil_threshold] += correct
            
        if majority == gt:
            print('{} was correctly classified by majority vote'.format(video_id))
            nb_correct += 1
        total += 1
    print('Nb correctly classified by majority: {}, out of {} videos'.format(
        nb_correct, total))
    for mil_threshold in mil_thresholds:
        print('\nNb correctly classified by MIL vote, threshold {}: {} out of {} videos'.format(
            mil_threshold, nb_correct_mil[mil_threshold], total))
        acc_mil = nb_correct_mil[mil_threshold]/total
        wandb.log({'video mil accuracy {}'.format(mil_threshold): acc_mil})
        print('Video level mil accuracy: ', acc_mil)
    acc = nb_correct/total
    wandb.log({'video level accuracy by majority vote': acc})
    print('Video level accuracy by majority vote: ', acc)


def evaluate_on_video_level(config_dict, model, model_path, test_dataset,
                            test_steps):

    test_acc_metric = tf.keras.metrics.BinaryAccuracy()
    model = model.model
    if config_dict['inference_only']:
        model.load_weights(model_path).expect_partial()
    else:
        model.load_weights(model_path)

    # @tf.function
    def test_step(x, preds, y):
        if config_dict['video_loss'] == 'cross_entropy':
            preds = model([x, preds], training=False)
        if config_dict['video_loss'] == 'mil':
            preds_seq = model([x, preds], training=True)
            preds_mil = evaluate_sparse_pain(y, preds_seq, config_dict)
            preds = preds_mil
        if config_dict['video_loss'] == 'mil_ce':
            preds_seq, preds_one = model([x, preds], training=True)
            preds_one = tf.keras.layers.Activation('softmax')(preds_one)
            preds_mil = evaluate_sparse_pain(y, preds_seq, config_dict)
            preds = 1/2 * (preds_one + preds_mil)
        y = y[:, 0, :]
        test_acc_metric.update_state(y, preds)
        return preds, y
    all_preds = []
    all_y = []
    with tqdm(total=test_steps) as pbar:
        for step, sample in enumerate(test_dataset):
            if step > test_steps:
                break
            pbar.update(1)
            # step_start_time = time.time()
            feats_batch, preds_batch, labels_batch, _ = sample
            preds, y = test_step(feats_batch, preds_batch, labels_batch)
            all_preds.append(preds)
            all_y.append(y)
    all_preds = make_array(all_preds)
    all_y = make_array(all_y)
    cm = confusion_matrix(tf.argmax(all_y, axis=1), tf.argmax(all_preds, axis=1))
    equality = tf.math.equal(tf.argmax(all_preds, axis=1), tf.argmax(all_y, axis=1))
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    print('\n Confusion matrix: \n', cm)
    test_acc = test_acc_metric.result()
    wandb.log({'test_acc': test_acc})
    print("Test acc keras metric: %.4f" % (float(test_acc),))
    test_acc = accuracy
    wandb.log({'test_acc_manual': test_acc})
    print("Test acc manual: %.4f" % (float(test_acc),))


def make_array(list_of_tensors):
    list_of_arrays = [lt.numpy() for lt in list_of_tensors]
    return np.concatenate(list_of_arrays)


def evaluate_sparse_pain(y_batch, preds_batch, config_dict):
    batch_size = y_batch.shape[0]  # last batch may be smaller
    kmax_scores = train.get_k_max_scores_per_class(y_batch, preds_batch, batch_size, config_dict)
    batch_class_distribution = tf.keras.layers.Activation('softmax')(kmax_scores)
    return batch_class_distribution


def run_evaluation(config_dict, model, model_path,
                   test_dataset, test_steps,
                   y_batches, y_paths):

    model = model.model
    if config_dict['inference_only']:
        model.load_weights(model_path).expect_partial()
    else:
        model.load_weights(model_path)

    ev = Evaluator(acc=True,
                   cm=True,
                   cr=True,
                   auc=True,
                   target_names=config_dict['target_names'],
                   batch_size=config_dict['batch_size'])

    y_preds = model.predict(x=test_dataset,
                            steps=test_steps,
                            verbose=1)

    nb_batches = y_batches.shape[0]

    y_test = np.reshape(y_batches, (nb_batches*config_dict['batch_size'],
                        config_dict['nb_labels']))
    confidence_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
    for confthresh in confidence_thresholds:
        print('\n CONFTHRESH: ', confthresh)
        majvotes = get_per_video_vote(
            y_pred=y_preds, y_paths=y_paths, ground_truth=y_test,
            confidence_threshold=confthresh)

        compute_video_level_accuracy(majvotes)

    # Take argmax of the probabilities.
    y_preds_argmax = np.argmax(y_preds, axis=1)
    y_preds_onehot = tf.keras.utils.to_categorical(y_preds_argmax,
                                                   num_classes=config_dict['nb_labels'])

    # Evaluate the model's performance
    ev.evaluate(model=model, y_test=y_test, y_pred=y_preds_onehot,
                softmax_predictions=y_preds,
                config_dict=config_dict, y_paths=y_paths)
