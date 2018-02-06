from image_processor import process_image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np


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


def random_crop(img_paths, width, height, crop_scale,
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
    X_crops = []
    tf.reset_default_graph()
    # offset_height = tf.random_uniform([], minval=0, maxval=0.2*height, seed=3)
    # offset_height = tf.cast(offset_height, tf.int32)
    # offset_width = tf.random_uniform([], minval=0, maxval=0.2*width, seed=3)
    # offset_width = tf.cast(offset_width, tf.int32)
    offset_height = tf.cast(crop_scale * height, tf.int32)
    offset_width = tf.cast(crop_scale * width, tf.int32)
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

if __name__ == '__main__':
    width = 128
    height = 128
    crop_width = 99
    crop_height = 99
    path_root = 'data/jpg_128_128_2fps/horse_4/4_1a/frame_0000'
    path = 'data/jpg_128_128_2fps/horse_4/4_1a/frame_000020.jpg'

    path_list = []

    for i in range(20,30):
        p = path_root + str(i) + '.jpg'
        path_list.append(p)

    random_scale_for_crop = np.random.rand()
    illuminance_scale = np.random.rand()
    crop_scale = random_scale_for_crop * 0.2
    illuminance_scale = np.random.uniform(0, 10)


    cropped_ims = random_crop(path_list, width, height,
                              crop_scale, crop_width, crop_height)
    flipped_ims = flip_images(path_list, width, height)
    adjusted_light_ims = adjust_contrast(path_list, illuminance_scale, width, height)

    rows = 5
    cols = 4
    f, axarr = plt.subplots(rows, cols)
    for i in range(0, rows):
        for j in range(0, cols):
            if j == 0:
                im = mpimg.imread(path_list[i])
                axarr[i, j].imshow(im)
            elif j == 1:
                im = cropped_ims[i]
                im /= 255
                axarr[i, j].imshow(im)
            elif j == 2:
                im = flipped_ims[i]
                im /= 255
                axarr[i, j].imshow(im)
            else:
                im = adjusted_light_ims[i]
                im /= 255
                axarr[i, j].imshow(im)
    plt.show()


