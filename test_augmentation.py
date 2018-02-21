from image_processor import process_image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import cv2


def get_image(path, width, height):
    im = mpimg.imread(path)
    # im = process_image(path, (width, height, 3))
    return im


def flip_images(img_paths, width, height):
    """
    Horizontal flip.
    :param images:
    :param width:
    :param height:
    :return:
    """
    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(width, height, 3))
    tf_img1 = tf.image.flip_left_right(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for index, img_path in enumerate(img_paths):
            img = mpimg.imread(img_path)
            flipped_imgs = sess.run([tf_img1], feed_dict={X: img})
            X_flip.extend(flipped_imgs)
    X_flip = np.array(X_flip, dtype=np.float32)

    return X_flip


def random_crop(img_paths, width, height, crop_scale_w, crop_scale_h,
                target_height, target_width):
    """
    Random crop but consistent across sequence.
    :param images:
    :param width:
    :param height:
    :param offset_height:
    :param offset_width:
    :param target_height:
    :param target_width:
    :return:
    """
    # Want both random offset height and width, separately.
    X_crops = []
    tf.reset_default_graph()
    offset_height = tf.cast(crop_scale_h * height, tf.int32)
    offset_width = tf.cast(crop_scale_w * width, tf.int32)

    X = tf.placeholder(tf.float32, shape=(width, height, 3))
    tf_img1 = tf.image.crop_to_bounding_box(X, offset_height, offset_width,
                                            target_height, target_width)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for index, img_path in enumerate(img_paths):
            img = mpimg.imread(img_path)
            cropped_imgs = sess.run([tf_img1], feed_dict={X:img})
            X_crops.extend(cropped_imgs)
    X_crops = np.array(X_crops, dtype=np.float32)
    return X_crops

def central_scale_images(X_imgs_paths, scales, offset_height, offset_width,
                         target_height, target_width):
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype = np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
    box_ind = np.zeros((len(scales)), dtype = np.int32)
    crop_size = np.array([height, width], dtype = np.int32)
    
    X_scale_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (1, height, width, 3))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for img_data_path in X_imgs_paths:
            img_data = mpimg.imread(img_data_path)
            batch_img = np.expand_dims(img_data, axis = 0)
            scaled_imgs = sess.run(tf_img, feed_dict = {X: batch_img})
            X_scale_data.extend(scaled_imgs)
    
    X_scale_data = np.array(X_scale_data, dtype = np.float32)
    return X_scale_data


def random_crop_resize(img_paths, width, height, target_height, target_width):
    """
    Random crop but consistent across sequence.
    :param images:
    :param width:
    :param height:
    :param offset_height:
    :param offset_width:
    :param target_height:
    :param target_width:
    :return:
    """
    random_scale_for_crop_w = np.random.rand()
    random_scale_for_crop_h = np.random.rand()

    crop_scale_w = random_scale_for_crop_w * 0.2
    crop_scale_h = random_scale_for_crop_h * 0.2
    print('Crop scale w and h: ', crop_scale_w, crop_scale_h)

    offset_height = crop_scale_h * height
    offset_width = crop_scale_w * width
    # y1 x1 are relative starting heights and widths in the crop box.
    # [[0, 0, 1, 1]] would mean no crop and just resize.

    y1 = offset_height/(height-1)
    x1 = offset_width/(width-1)
    y2 = (offset_height + target_height)/(height-1)
    x2 = (offset_width + target_width)/(width-1)

    boxes = np.array([[y1, x1, y2, x2]], dtype=np.float32)
    box_ind = np.array([0], dtype=np.int32)
    crop_size = np.array([height, width], dtype=np.int32)

    X_crops = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(1, width, height, 3))
    tf_img1 = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for index, img_path in enumerate(img_paths):
            img = mpimg.imread(img_path)
            batch_img = np.expand_dims(img, axis = 0)
            cropped_imgs = sess.run([tf_img1], feed_dict={X: batch_img})
            X_crops.extend(cropped_imgs)
    X_crops = np.array(X_crops, dtype=np.float32)
    X_crops = np.reshape(X_crops, (10, 128, 128, 3))
    return X_crops


def adjust_illuminance(img_paths, illuminance_scale, width, height):
    X_adjusted = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(width, height, 3))
    tf_img1 = tf.image.adjust_brightness(X, delta=illuminance_scale)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for index, img_path in enumerate(img_paths):
            img = mpimg.imread(img_path)
            flipped_imgs = sess.run([tf_img1], feed_dict={X: img})
            X_adjusted.extend(flipped_imgs)
    X_adjusted = np.array(X_adjusted, dtype=np.float32)

    return X_adjusted


def adjust_contrast(img_paths, scale, width, height):
    X_adjusted = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(width, height, 3))
    tf_img1 = tf.image.adjust_contrast(X, scale)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for index, img_path in enumerate(img_paths):
            img = mpimg.imread(img_path)
            flipped_imgs = sess.run([tf_img1], feed_dict={X: img})
            X_adjusted.extend(flipped_imgs)
    X_adjusted = np.array(X_adjusted, dtype=np.float32)

    return X_adjusted

def gaussian_noise_addition(input_tensor, std):
    noise = tf.random_normal(shape=tf.shape(input_tensor), mean=0.0, stddev=std, dtype=tf.float32)
    return input_tensor + noise


def adjust_lighting_by_gn(img_paths, std, width, height):
    X_adjusted = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(width, height, 3))
    tf_img1 = gaussian_noise_addition(X, std)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for index, img_path in enumerate(img_paths):
            img = mpimg.imread(img_path)
            flipped_imgs = sess.run([tf_img1], feed_dict={X: img})
            X_adjusted.extend(flipped_imgs)
    X_adjusted = np.array(X_adjusted, dtype=np.float32)

    return X_adjusted

def add_gaussian_noise(img_paths, im_weight, noise_weight, width, height):
    gaussian_noise_imgs = []

    mean = 0
    sigma = 0.5

    row, col, ch = height, width, 3

    gaussian = np.random.normal(mean, sigma, (row, col, ch)).astype(np.float32)

    imw_a = 0.4
    imw_b = 0.65
    im_weight = (imw_b - imw_a) * np.random.random() + imw_a
    
    now_a = 0.2
    now_b = 0.5
    noise_weight = (now_b - now_a) * np.random.random() + now_a

    print('Image weight: {}, noise weight: {}', im_weight, noise_weight)
    for index, img_path in enumerate(img_paths):
        img = mpimg.imread(img_path).astype(np.float32)

        gaussian_img = cv2.addWeighted(img, im_weight, gaussian, noise_weight, 0)
        gaussian_noise_imgs.append(gaussian_img)

    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype=np.float32)
    return gaussian_noise_imgs




def adjust_lighting(img_paths, delta, width, height):
    X_adjusted = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(width, height, 3))
    tf_img1 = tf.image.adjust_brightness(X, delta)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for index, img_path in enumerate(img_paths):
            img = mpimg.imread(img_path)
            transformed_imgs = sess.run([tf_img1], feed_dict={X: img})
            X_adjusted.extend(transformed_imgs)
    X_adjusted = np.array(X_adjusted, dtype=np.float32)

    return X_adjusted


if __name__ == '__main__':
    width = 128
    height = 128
    crop_width = 99
    crop_height = 99
    path_root = 'data/jpg_128_128_2fps/horse_3/3_1c/frame_0000'
    path = 'data/jpg_128_128_2fps/horse_3/3_1c/frame_000020.jpg'

    path_list = []

    for i in range(80,90):
        p = path_root + str(i) + '.jpg'
        path_list.append(p)

    illuminance_scale = np.random.uniform(0, 1)
    brightness_delta = np.random.uniform(0, 0.2)
    brightness_delta = 0.7
    std_gaussian_noise = np.random.uniform(0.2,0.5)    
    std_gaussian_noise = 7
    print('Contrast adjustment scale: ', illuminance_scale)
    print('Gaussian noise std: ', std_gaussian_noise)
    print('Brightness delta: ', brightness_delta)

    # cropped_ims1 = central_scale_images(path_list, [0.80], offset_height, offset_width, crop_height, crop_width)
    # cropped_ims2 = central_scale_images(path_list, [0.80], offset_height, offset_width, crop_height, crop_width)
    # cropped_ims3 = central_scale_images(path_list, [0.80], offset_height, offset_width, crop_height, crop_width)
    # cropped_ims4 = central_scale_images(path_list, [0.80], offset_height, offset_width, crop_height, crop_width)
    cropped_ims1 = random_crop_resize(path_list, width, height, crop_width, crop_height)

    cropped_ims2 = random_crop_resize(path_list, width, height, crop_width, crop_height)

    cropped_ims3 = random_crop_resize(path_list, width, height, crop_width, crop_height)

    cropped_ims4 = random_crop_resize(path_list, width, height, crop_width, crop_height)

    flipped_ims = flip_images(path_list, width, height)
    adjusted_contrast_ims = adjust_contrast(path_list, illuminance_scale, width, height)
    # gn_adjusted_light_ims = adjust_lighting_by_gn(path_list, std_gaussian_noise, width, height)
    mean = 0
    gn_adjusted_light_ims = add_gaussian_noise(path_list, 0.25, 0.25 , width, height)
    gn_adjusted_light_ims2 = add_gaussian_noise(path_list, 0.50, 0.5 , width, height)
    gn_adjusted_light_ims3 = add_gaussian_noise(path_list, 0.6, 0.4, width, height)
    gn_adjusted_light_ims4 = add_gaussian_noise(path_list, 0.9, 0.1, width, height)
    adjusted_light_ims = adjust_lighting(path_list, brightness_delta, width, height)

    rows = 5
    cols = 5
    f, axarr = plt.subplots(rows, cols, figsize=(20,10))
    # for i in range(0, rows):
    #     for j in range(0, cols):
    #         if j == 0:
    #             im = mpimg.imread(path_list[i])
    #             axarr[i, j].imshow(im)
    #         elif j == 1:
    #             im = cropped_ims[i]
    #             im /= 255
    #             axarr[i, j].imshow(im)
    #         elif j == 2:
    #             im = flipped_ims[i]
    #             im /= 255
    #             axarr[i, j].imshow(im)
    #         elif j == 3:
    #             im = adjusted_contrast_ims[i]
    #             im /= 255
    #             axarr[i, j].imshow(im)
    #         else:
    #             im = gn_adjusted_light_ims[i]
    #             im /= 255
    #             axarr[i, j].imshow(im)
    # for i in range(0, rows):
    #     for j in range(0, cols):
    #         if j == 0:
    #             im = mpimg.imread(path_list[i])
    #             axarr[i, j].imshow(im)
    #         elif j == 1:
    #             im = gn_adjusted_light_ims[i]
    #             im /= 255
    #             axarr[i, j].imshow(im)
    #         elif j == 2:
    #             im = gn_adjusted_light_ims2[i]
    #             im /= 255
    #             axarr[i, j].imshow(im)
    #         elif j == 3:
    #             im = gn_adjusted_light_ims3[i]
    #             im /= 255
    #             axarr[i, j].imshow(im)
    #         else:
    #             im = gn_adjusted_light_ims4[i]
    #             im /= 255
    #             axarr[i, j].imshow(im)
    for i in range(0, rows):
        for j in range(0, cols):
            if j == 0:
                im = mpimg.imread(path_list[i])
                axarr[i, j].imshow(im)
            elif j == 1:
                im = cropped_ims1[i]
                im /= 255
                axarr[i, j].imshow(im)
            elif j == 2:
                im = cropped_ims2[i]
                im /= 255
                axarr[i, j].imshow(im)
            elif j == 3:
                im = cropped_ims3[i]
                im /= 255
                axarr[i, j].imshow(im)
            else:
                im = cropped_ims4[i]
                im /= 255
                axarr[i, j].imshow(im)
    plt.show()

