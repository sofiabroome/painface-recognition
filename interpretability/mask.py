import tensorflow as tf
import numpy as np

from tf_slice_assign import slice_assign

MASK_THRESHOLD = 0.1


def find_submasks_from_mask(mask, threshold=MASK_THRESHOLD):
    """
    :param mask:
    :param threshold: float
    :return: [[int]], list of list of mask elements
    """
    print(type(mask))
    submasks = []
    current_submask = []
    currently_in_mask = False
    for j in range(len(mask)):
        # if we find a value above threshold but is first occurence, start appending to current submask
        if mask[j] > threshold and not currently_in_mask:
            current_submask = []
            currently_in_mask = True
            current_submask.append(j)
        # if it's not current occurence, just keep appending
        elif mask[j] > threshold and currently_in_mask:
            current_submask.append(j)
            # if below thresh, stop appending
        elif mask[j] <= threshold and currently_in_mask:
            submasks.append(current_submask)
            currently_in_mask = False
        # if at the end of clip, create last submask
        if j == len(mask) - 1 and currently_in_mask:
            submasks.append(current_submask)
            currently_in_mask = False
    print("submasks found: ", submasks)
    return submasks


def perturb_sequence(seq, mask, perturbation_type='freeze', snap_values=False):
    if snap_values:
        for j in range(len(mask)):
            if mask[j] > 0.5:
                mask[j] = 1
            else:
                mask[j] = 0
    if perturbation_type == 'freeze':
        perturbed_seq = tf.zeros(seq.shape)
        for u in range(len(mask)):

            if u == 0:
                # tf does not support slice assign of tensors.
                # the below is eq. to perturbed_seq[:, u, ...] = seq[:, u, ...]
                perturbed_seq = slice_assign(
                    perturbed_seq,
                    tf.expand_dims(seq[:, u, :, :, :], axis=0),
                    ':', slice(u, u + 1, 1), ':', ':', ':')
            if u != 0:
                to_assign = (1 - mask[u]) * seq[:, u, :, :, :] + mask[u] * perturbed_seq[:, u - 1, :, :, :]
                perturbed_seq = slice_assign(
                    perturbed_seq,
                    tf.expand_dims(to_assign, axis=0),
                    ':', slice(u, u + 1, 1), ':', ':', ':')

    if perturbation_type == 'reverse':
        perturbed_seq = tf.zeros(seq.shape)

        # # threshold for finding out which indexes are on
        # maskOnInds = (mask > 0.1).nonzero()
        # if len(maskOnInds) > 0:
        #     # non-zero gives unfortunate dimensions like [[0], [1]] so squeeze it
        #     maskOnInds = maskOnInds.squeeze(dim=1)
        # maskOnInds = maskOnInds.tolist()

        # submasks = findSubMasksFromMask(mask, 0.1)
        submasks = find_submasks_from_mask(mask, 0.1)

        for y in range(len(mask)):
            # Start from the original sequence
            perturbed_seq = slice_assign(
                ':', slice(y, y + 1, 1), ':', ':', ':',
                sliced_tensor=perturbed_seq,
                assigned_tensor=tf.expand_dims(seq[:, y, :, :, :], axis=0))

        for maskOnInds in submasks:
            '''if the submask is on at 3,4,5,6,7. the point is to go to the almost halfway point
            as in 3,4 (so that's why it's len()//2) and swap it with the u:th last element instead (like 7,6)
            '''
            for u in range(int(len(maskOnInds) // 2)):
                # temp storage when doing swap
                temp = seq[:, :, maskOnInds[u], :, :].clone()
                # first move u:th LAST frame to u:th frame
                perturbed_seq[:, :, maskOnInds[u], :, :] = (1 - mask[maskOnInds[u]]) * seq[:, :, maskOnInds[u], :, :] + \
                                                       mask[maskOnInds[u]] * seq[:, :, maskOnInds[-(u + 1)], :, :]

                # then use temp storage to move u:th frame to u:th last frame
                perturbed_seq[:, :, maskOnInds[-(u + 1)], :, :] = (1 - mask[maskOnInds[u]]) * seq[:, :,
                                                                                          maskOnInds[-(u + 1)], :, :] + \
                                                              mask[maskOnInds[u]] * temp
                # print("return type of pertb: ", perturbed_seq.type())
    return perturbed_seq


def init_mask(seq, target, model, thresh=0.9, mode="central"):
    """
    Initiaizes the first value of the mask where the gradient descent
    methods for finding the masks starts. Central finds the smallest
    centered mask which still reduces the class score by at least 90%
    compared to a fully perturbing mask (whole mask on). Random just
    turns (on average) 70% of the mask
    (Does not give very conclusive results so far).
    """
    if mode == 'central':

        # get the class score for the fully perturbed sequence
        full_pert_score = model(perturb_sequence(seq, np.ones(seq.shape[1], dtype='float32')))
        full_pert_score = full_pert_score[:, np.argmax(target)]

        orig_score = model(perturb_sequence(seq, np.zeros((seq.shape[0]))))
        orig_score = orig_score[:, np.argmax(target)]

        # reduce mask size while the loss ratio remains above 90%
        for i in range(1, seq.shape[1] // 2):
            new_mask = np.ones(seq.shape[1])
            new_mask[:i] = 0
            new_mask[-i:] = 0

            central_score = model(perturb_sequence(seq, new_mask))
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


def calc_TV_norm(mask, config_dict):
    """"
    Calculates the Total Variational Norm by summing the differences of the values
    in between the different positions in the mask.
    p=3 and q=3 are defaults from the paper.
    """
    val = 0
    for u in range(1, config_dict['seq_length'] - 1):
        val += tf.abs(mask[u - 1] - mask[u]) ** config_dict['tv_norm_p']
        val += tf.abs(mask[u + 1] - mask[u]) ** config_dict['tv_norm_p']
    val = val ** (1 / config_dict['tv_norm_p'])
    val = val ** config_dict['tv_norm_q']

    return val
