import tensorflow as tf
import numpy as np

from tf_slice_assign import slice_assign

MASK_THRESHOLD = 0.1


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


def perturbSequence(seq, mask, perbType='freeze', snapValues=False):
    if (snapValues):
        for j in range(len(mask)):
            if (mask[j] > 0.5):
                mask[j] = 1
            else:
                mask[j] = 0
    if (perbType == 'freeze'):
        # pytorch expects Batch,Channel, T, H, W
        perbInput = tf.zeros(seq.shape)  # seq.clone().detach()
        # TODO: also double check that the mask here is not a copy but by ref so the grad holds
        for u in range(len(mask)):

            if (u == 0):
                perbInput = slice_assign(perbInput,
                                         tf.expand_dims(seq[:, u, :, :, :], axis=0),
                                         ':', slice(u, u+1, 1), ':', ':', ':')
                print('pi shape:', perbInput.shape)
                # perbInput[:, :, u, :, :] = seq[:, :, u, :, :]
            if (u != 0):
                to_assign = (1 - mask[u]) * seq[:, u, :, :, :] + mask[u] * perbInput.numpy()[:, u-1, :, :, :]
                perbInput = slice_assign(perbInput, tf.expand_dims(to_assign, axis=0),
                                         ':', slice(u, u + 1, 1), ':', ':', ':')

    if (perbType == 'reverse'):
        # pytorch expects Batch,Channel, T, H, W
        perbInput = tf.zeros(seq.shape)

        # threshold for finding out which indexes are on
        maskOnInds = (mask > 0.1).nonzero()
        if (len(maskOnInds) > 0):
            # non-zero gives unfortunate dimensions like [[0], [1]] so squeeze it
            maskOnInds = maskOnInds.squeeze(dim=1)
        maskOnInds = maskOnInds.tolist()

        # subMasks = findSubMasksFromMask(mask, 0.1)
        subMasks = find_submasks_from_mask(mask, 0.1)

        for y in range(len(mask)):
            # start with original
            perbInput[:, :, y, :, :] = seq[:, :, y, :, :]

        for maskOnInds in subMasks:
            '''if the submask is on at 3,4,5,6,7. the point is to go to the almost halfway point
            as in 3,4 (so that's why it's len()//2) and swap it with the u:th last element instead (like 7,6)
            '''
            for u in range(int(len(maskOnInds) // 2)):
                # temp storage when doing swap
                temp = seq[:, :, maskOnInds[u], :, :].clone()
                # first move u:th LAST frame to u:th frame
                perbInput[:, :, maskOnInds[u], :, :] = (1 - mask[maskOnInds[u]]) * seq[:, :, maskOnInds[u], :, :] + \
                                                       mask[maskOnInds[u]] * seq[:, :, maskOnInds[-(u + 1)], :, :]

                # then use temp storage to move u:th frame to u:th last frame
                perbInput[:, :, maskOnInds[-(u + 1)], :, :] = (1 - mask[maskOnInds[u]]) * seq[:, :,
                                                                                          maskOnInds[-(u + 1)], :, :] + \
                                                              mask[maskOnInds[u]] * temp
                # print("return type of pertb: ", perbInput.type())
    return perbInput


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


def init_mask(seq, target, model, forward_fn, thresh=0.9, mode="central"):
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
        print('types ', seq.dtype, np.ones(seq.shape[1]).dtype)
        # full_pert_score = forward_fn(seq, np.ones(seq.shape[1], dtype='float32'))
        full_pert_score = model(perturbSequence(seq, np.ones(seq.shape[1], dtype='float32')))
        print(full_pert_score.shape)

        full_pert_score = full_pert_score[:, np.argmax(target)]
        print(full_pert_score.shape)

        # orig_score = forward_fn(seq, np.zeros((seq.shape[0])))
        orig_score = model(perturbSequence(seq, np.zeros((seq.shape[0]))))
        print(orig_score.shape)
        orig_score = orig_score[:, np.argmax(target)]
        print(orig_score.shape)

        # reduce mask size while the loss ratio remains above 90%
        for i in range(1, seq.shape[1] // 2):
            new_mask = np.ones(seq.shape[1])
            new_mask[:i] = 0
            new_mask[-i:] = 0

            # central_score = forward_fn(seq, new_mask)
            central_score = model(perturbSequence(seq, new_mask))
            print(central_score.shape)
            central_score = central_score[:, np.argmax(target)]
            print(central_score.shape)

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
