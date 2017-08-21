from keras.preprocessing.image import load_img, img_to_array
import numpy as np


def process_image(image, target_shape):
    """
    Process and image and return the corresponding numerical array.
    :param image: image
    :param target_shape: (int, int, int)
    :return: np.ndarray
    """
    # Load the image
    h, w, _ = target_shape
    img = load_img(image, target_size=target_shape)
    return img_to_array(img).astype(np.float32)
