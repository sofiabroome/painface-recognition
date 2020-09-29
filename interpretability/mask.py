import tensorflow as tf
import numpy as np

MASK_THRESHOLD = 0.1
FLAGS = tf.app.flags.FLAGS


def find_submasks_from_mask(mask, thresh=MASK_THRESHOLD):
    submasks = []
    current_submask = []
    currently_in_mask = False
    for j in range(len(mask)):
        # if we find a value above threshold but is first occurence, start appending to current submask
        if mask[j] > thresh and not currently_in_mask:
            current_submask = []
            currently_in_mask = True
            current_submask.append(j)
        # if it's not current occurence, just keep appending
        elif mask[j] > thresh and currently_in_mask:
            current_submask.append(j)
            # if below thresh, stop appending
        elif mask[j] <= thresh and currently_in_mask:
            submasks.append(current_submask)
            currently_in_mask = False
        # if at the end of clip, create last submask
        if j == len(mask) - 1 and currently_in_mask:
            submasks.append(current_submask)
            currently_in_mask = False
    # print("submasks found: ", submasks)
    return submasks


def perturb_sequence(seq, mask, perb_type='freeze', snap_values=False):
    if snap_values:
        for j in range(len(mask)):
            if mask[j] > 0.5:
                mask[j] = 1
            else:
                mask[j] = 0

    if perb_type == 'freeze':
        perb_input = np.zeros(seq.shape)
        for u in range(len(mask)):

            if u == 0:  # Set first frame to same as seq.
                perb_input[:, u, :, :, :] = seq[:, u, :, :, :]

            if u != 0:  # mask[u]>=0.5 and u!=0
                perb_input[:, u, :, :, :] = (1 - mask[u]) * seq[:, u, :, :, :] + \
                                           mask[u] * np.copy(perb_input)[:, u - 1, :, :, :]

    if perb_type == 'reverse':
        perb_input = np.zeros(seq.shape)
        mask_on_inds = np.where(mask > MASK_THRESHOLD)[0]  # np.where returns a tuple for some reason
        mask_on_inds = mask_on_inds.tolist()

        sub_masks = find_submasks_from_mask(mask)

        for y in range(len(mask)):
            perb_input[:, y, :, :, :] = seq[:, y, :, :, :]

        for mask_on_inds in sub_masks:
            # leave unmasked parts alone (as well as reverse middle point)
            if (len(mask_on_inds) // 2 < len(mask_on_inds) / 2) and y == mask_on_inds[(len(mask_on_inds) // 2)]:
                # print("hit center at ", y)
                perb_input[:, y, :, :, :] = seq[:, y, :, :, :]
            for u in range(int(len(mask_on_inds) // 2)):
                temp = seq[:, mask_on_inds[u], :, :, :]
                perb_input[:, mask_on_inds[u], :, :, :] = (1 - mask[mask_on_inds[u]]) *\
                    seq[:, mask_on_inds[u], :, :, :] + \
                    mask[mask_on_inds[u]] *\
                    seq[:, mask_on_inds[-(u + 1)], :, :, :]
                perb_input[:, mask_on_inds[-(u + 1)], :, :, :] = (1 - mask[mask_on_inds[u]]) *\
                    seq[:, mask_on_inds[-(u + 1)], :, :, :] +\
                    mask[mask_on_inds[ u]] * temp
    return perb_input


def init_mask(seq, original_input_var, frame_inds, after_softmax, target,
              thresh=0.9, mode="central", forward_fn=forward_fn,
              mask_pert_type='freeze'):
    """
    Initiaizes the first value of the mask where the gradient descent methods for finding
    the masks starts. Central finds the smallest centered mask which still reduces the class score by 
    at least 90% compared to a fully perturbing mask (whole mask on). Random just turns (on average) 70% of the 
    mask (Does not give very conclusive results so far). 
    """
    if mode == 'central':

        # first define the fully perturbed sequence
        full_pert = np.zeros(seq.shape)
        for i in range(seq.shape[1]):
            full_pert[:, i, :, :, :] = seq[:, 0, :, :, :]

        # get the class score for the fully perturbed sequence
        full_pert_score = sess.run(after_softmax, feed_dict={mask_var: np.ones((FLAGS.seq_length)),
                                                             original_input_var: seq,
                                                             frame_inds: range(FLAGS.seq_length)})
        full_pert_score = full_pert_score[:, np.argmax(target)]

        orig_score = sess.run(after_softmax, feed_dict={mask_var: np.zeros((FLAGS.seq_length)),
                                                        original_input_var: seq,
                                                        frame_inds: range(FLAGS.seq_length)})
        orig_score = orig_score[:, np.argmax(target)]

        # reduce mask size while the loss ratio remains above 90%
        for i in range(1, seq.shape[1] // 2):
            new_mask = np.ones(seq.shape[1])
            new_mask[:i] = 0
            new_mask[-i:] = 0

            central_score = sess.run(after_softmax, feed_dict={mask_var: new_mask,
                                                               original_input_var: seq,
                                                               frame_inds: range(FLAGS.seq_length)})
            central_score = central_score[:, np.argmax(target)]

            score_ratio = (orig_score - central_score) / (orig_score - full_pert_score)

            if score_ratio < thresh:
                break

        mask = new_mask

        # modify the mask so that it is roughly 0 or 1 after sigmoid
        for j in range(len(mask)):
            if mask[j] == 0:
                mask[j] = -5
            elif mask[j] == 1:
                mask[j] = 5

    print('Initial mask: ', mask)
    return mask


def calc_TV_norm(mask, p=3, q=3):
    """"
    Calculates the Total Variational Norm by summing the differences of the values
    in between the different positions in the mask.
    p=3 and q=3 are defaults from the paper.
    """
    val = 0
    for u in range(1, FLAGS.seq_length - 1):
        val += tf.abs(mask[u - 1] - mask[u]) ** p
        val += tf.abs(mask[u + 1] - mask[u]) ** p
    val = val ** (1 / p)
    val = val ** q

    return val
