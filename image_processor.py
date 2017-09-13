from keras.preprocessing.image import load_img, img_to_array
import numpy as np


def process_image(image_path, target_shape):
    """
    Process and image and return the corresponding numerical array.
    :param image_path: str
    :param target_shape: (int, int, int)
    :return: np.ndarray
    """
    # Load the image
    w, h, c = target_shape
    target_shape = (h, w, c)
    img = load_img(image_path, target_size=target_shape)

    return img_to_array(img).astype(np.float32)
