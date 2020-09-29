import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from interpretability import mask

FLAGS = tf.app.flags.FLAGS


def visualize_results(orig_seq, pert_seq, mask, root_dir=None,
                      case="0", mark_imgs=True, iter_test=False):
    if root_dir is None:
        root_dir = '/workspace/projects/spatiotemporal-interpretability/tensorflow/' + \
                   FLAGS.output_folder + "/"
    root_dir += "/PerturbImgs/"

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for i in range(orig_seq.shape[1]):

        if mark_imgs:
            orig_seq[:, i, :10, :10, 1:] = 0
            orig_seq[:, i, :10, :10, 0] = mask[i] * 255
            pert_seq[:, i, :10, :10, 1:] = 0
            pert_seq[:, i, :10, :10, 0] = mask[i] * 255
        result = Image.fromarray(pert_seq[0, i, :, :, :].astype(np.uint8))
        result.save(root_dir + "case" + case + "pert" + str(i) + ".png")
    f = open(root_dir + "case" + case + ".txt", "w+")
    f.write(str(mask))
    f.close()


def visualize_results_on_gradcam(gradcam_images, mask, root_dir,
                                 image_width, image_height,
                                 case="0", round_up_mask=True):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    dots = find_temp_mask_red_dots(image_width, image_height, mask, round_up_mask)

    dot_offset = FLAGS.image_width * 2
    for i in range(len(mask)):
        for j, dot in enumerate(dots):

            if i == j:
                intensity = 255
            else:
                intensity = 150

            gradcam_images[i][dot["y_start"]:, dot_offset + dot["x_start"]:dot_offset + dot["x_end"], :] = 0
            gradcam_images[i][
                dot["y_start"]:,
                dot_offset + dot["x_start"]:dot_offset + dot["x_end"],
                dot["channel"]] = intensity

            # result = Image.fromarray(gradcam_images[i].astype(np.uint8), mode="RGB")
            tmp = cv2.cvtColor(gradcam_images[i], cv2.COLOR_BGR2RGB)
            result = Image.fromarray(tmp.astype(np.uint8))
            result.save(root_dir + "/case" + case + "_" + str(i) + ".png")

    f = open(root_dir + "/MASKVALScase" + case + ".txt", "w+")
    f.write(str(mask))
    f.close()


def find_temp_mask_red_dots(image_width, image_height, mask, round_up_mask):
    mask_len = len(mask)
    dot_width = int(image_width // (mask_len + 4))
    dot_padding = int((image_width - (dot_width * mask_len)) // mask_len)
    dot_height = int(image_height // 20)
    dots = []

    for i, m in enumerate(mask):

        if round_up_mask:
            if mask[i] > 0.5:
                mask[i] = 1
            else:
                mask[i] = 0

        dot = {'y_start': -dot_height,
               'y_end': image_height,
               'x_start': i * (dot_width + dot_padding),
               'x_end': i * (dot_width + dot_padding) + dot_width}

        if mask[i] == 0:
            dot['channel'] = 1  # Green
        else:
            dot['channel'] = 2  # in BGR.

        dots.append(dot)

    return dots


def create_image_arrays(input_sequence, gradcams, time_mask,
                        output_folder, video_id, mask_type,
                        image_width, image_height):
    combined_images = []
    for i in range(FLAGS.seq_length):
        input_data_img = input_sequence[0, i, :, :, :]

        time_mask_copy = time_mask.copy()

        combined_img = np.concatenate((np.uint8(input_data_img),
                                       np.uint8(gradcams[i]),
                                       np.uint8(mask.perturb_sequence(
                                           input_sequence,
                                           time_mask_copy,
                                           perb_type=mask_type,
                                           snap_values=True)[0, i, :, :, :])),
                                      axis=1)

        combined_images.append(combined_img)
        cv2.imwrite(os.path.join(
            output_folder,
            "img%02d.jpg" % (i + 1)),
            combined_img)

    visualize_results_on_gradcam(combined_images,
                                 time_mask,
                                 image_width,
                                 image_height,
                                 root_dir=output_folder,
                                 case=mask_type + video_id)

    return combined_images
