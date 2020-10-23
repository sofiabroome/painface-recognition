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
    # submasks = []
    # current_submask = []
    # Any Python side-effects (appending to a list, printing with print, etc)
    # will only happen once, when func is traced. To have side-effects
    # executed into your tf.function they need to be written as TF ops:
    submasks = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    current_submask = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    currently_in_mask = False
    submasks_append_counter = 0
    for j in range(len(mask)):
        # if we find a value above threshold but is first occurence, start appending to current submask
        if mask[j] > threshold and not currently_in_mask:
            # current_submask = []
            current_submask = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
            currently_in_mask = True
            # current_submask.append(j)
            current_submask.write(j, j)
            # current_submask.stack()
        # if it's not current occurence, just keep appending
        elif mask[j] > threshold and currently_in_mask:
            # current_submask.append(j)
            current_submask.write(j, j)
            # current_submask.stack()
            # if below thresh, stop appending
        elif mask[j] <= threshold and currently_in_mask:
            submasks.write(submasks_append_counter, current_submask.stack())
            submasks_append_counter += 1
            # submasks.append(current_submask)
            currently_in_mask = False
        # if at the end of clip, create last submask
        if j == len(mask) - 1 and currently_in_mask:
            submasks.write(submasks_append_counter, current_submask.stack())
            submasks_append_counter += 1
            # submasks.append(current_submask)
            currently_in_mask = False

    # submasks.stack()
    # print("submasks found: ", submasks)
    return submasks.stack()


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
        print('Reverse mask perturbation not supported in tf2.')
        raise NotImplementedError
        perturbed_seq = tf.zeros(seq.shape)

        # sub_masks_ta = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

        # def get_submasks():
        #     submasks = sub_masks_ta.write(0, find_submasks_from_mask(mask, 0.1))
        #     return submasks.stack()

        # submasks = get_submasks()
        submasks = find_submasks_from_mask(mask, 0.1)

        for y in range(len(mask)):
            # Start from the original sequence
            perturbed_seq = slice_assign(
                perturbed_seq,
                tf.expand_dims(seq[:, y, :, :, :], axis=0),
                ':', slice(y, y + 1, 1), ':', ':', ':')

        moi_ta = tf.TensorArray(tf.int32, size=0, dynamic_size=0)
        for moi in submasks:
            '''if the submask is on at 3,4,5,6,7. the point is to go to the almost halfway point
            as in 3,4 (so that's why it's len()//2) and swap it with the u:th last element instead (like 7,6)
            '''
            moi_ta.write(0, moi)
            mask_on_inds = moi_ta.stack()
            for u in range(int(len(mask_on_inds) // 2)):
                submask_index = mask_on_inds[u]
                mirror_submask_index = mask_on_inds[-(u+1)]
                # Temp storage when doing swap
                temp = seq[:, submask_index, :, :, :]
                # First move u:th LAST frame to u:th frame
                terms_to_assign = (1 - mask[submask_index]) * seq[:, submask_index, :, :, :] +\
                    mask[submask_index] * seq[:, mirror_submask_index, :, :, :]
                perturbed_seq = slice_assign(
                    perturbed_seq,
                    tf.expand_dims(terms_to_assign, axis=0),
                    ':', slice(submask_index, submask_index + 1, 1), ':', ':', ':'
                )

                # then use temp storage to move u:th frame to u:th last frame
                terms_to_assign = (1 - mask[submask_index]) * seq[:, mirror_submask_index, :, :, :] +\
                    mask[submask_index] * temp
                perturbed_seq = slice_assign(
                    perturbed_seq,
                    tf.expand_dims(terms_to_assign, axis=0),
                    ':', slice(mirror_submask_index, mirror_submask_index + 1, 1), ':', ':', ':'
                )
                # print("return type of pertb: ", perturbed_seq.type())
    return perturbed_seq


def init_mask(seq, target, model, config_dict, thresh=0.9, mode="central"):
    """
    Initiaizes the first value of the mask where the gradient descent
    methods for finding the masks starts. Central finds the smallest
    centered mask which still reduces the class score by at least 90%
    compared to a fully perturbing mask (whole mask on). Random just
    turns (on average) 70% of the mask
    (Does not give very conclusive results so far).
    """
    if mode == 'central':

        def perturb_one_or_two_streams(x, mask):
            if config_dict['model'] == '2stream_5d_add':
                perturbed_rgb = perturb_sequence(x[0, :], mask)
                perturbed_flow = perturb_sequence(x[1, :], mask)
                concat_streams = tf.concat([perturbed_rgb, perturbed_flow], axis=0)
                concat_streams_6d = tf.expand_dims(concat_streams, axis=1)
                print('concat streams shape: ', concat_streams.shape)
                score, _ = model(concat_streams_6d)
            else:
                score, _ = model(perturb_sequence(x, mask))
            return score

        # get the class score for the fully perturbed sequence
        full_pert_score = perturb_one_or_two_streams(seq, np.ones(config_dict['seq_length'], dtype='float32'))
        full_pert_score = full_pert_score[:, np.argmax(target)]

        orig_score, _ = model(seq)
        orig_score = orig_score[:, np.argmax(target)]

        # reduce mask size while the loss ratio remains above 90%
        for i in range(1, config_dict['seq_length'] // 2):
            new_mask = np.ones(config_dict['seq_length'])
            new_mask[:i] = 0
            new_mask[-i:] = 0

            central_score = perturb_one_or_two_streams(seq, new_mask)
            central_score = central_score[:, np.argmax(target)]
            print('\nReducing mask size, index: ', i)
            print('full pert score: ', full_pert_score)
            print('original score: ', orig_score)
            print('central score: ', central_score)

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
