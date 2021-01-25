from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import importlib
import os


def process_image(image_path, target_shape, standardize=True, mean=None, std=None, normalize=True):
    """
    Process and image and return the corresponding numerical array.
    :param image_path: str
    :param target_shape: (int, int, int)
    :param standardize: bool
    :param mean: [float, float, float] dataset pixel mean
    :param std: [float, float, float] dataset pixel std
    :return: np.ndarray
    """
    w, h, c = target_shape
    target_shape = (h, w, c)
    img = load_img(image_path, target_size=target_shape)
    img = img_to_array(img)
    if normalize:
        img = img/255
    if standardize:
        img = (img-mean)/std

    return img.astype(np.float32)


def load_module(module_path_and_name):
    # if contained in module it would be a oneliner:
    # config_dict_module = importlib.import_module(dict_module_name)
    module_child_name = module_path_and_name.split('/')[-1].replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_child_name, module_path_and_name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def flatten_batch_lists(batch_list, nb_batches):
    flat_list = []
    for b in range(nb_batches):
        flat_list += batch_list[b]
    return flat_list


def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path);


def get_last_characters_from_string(string, nb_chars):
    return string[-nb_chars:]


def split_string_at_last_occurence_of_certain_char(string, char):
    left, right = string.rsplit(sep=char, maxsplit=1)
    return left, right


def find_between(s, first, last):
    """
    Find string between two given strings. First occurence.
    :param s: whole string [str]
    :param first: first string [str]
    :param last: last string [str]
    :return: the string in between [str]
    """
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""


def write_file(file_name, list_to_write):
    with open(file_name, 'w') as f:
        for string in list_to_write:
            f.write(string+'\n');


def plot_augmentation(train, val, test, rgb, X_seq_list, flipped, cropped, shaded,
                      seq_index, batch_index, window_index):
    rows = 4
    cols = 10
    f, axarr = plt.subplots(rows, cols, figsize=(20, 10))
    for i in range(0, rows):
        for j in range(0, cols):
            axarr[i, j].set_xticks([])
            axarr[i, j].set_yticks([])
            if i == 0:
                im = X_seq_list[j]
                im /= 255
                axarr[i, j].imshow(im)
            elif i == 1:
                im = flipped[j]
                im /= 255
                axarr[i, j].imshow(im)
            elif i == 2:
                im = cropped[j]
                im /= 255
                axarr[i, j].imshow(im)
            else:
                im = shaded[j]
                im /= 255
                axarr[i, j].imshow(im)
    plt.tick_params(axis='both', which='both', bottom='off', left='off')
    f.subplots_adjust(wspace=0, hspace=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    if train:
        partition = 1
    elif val:
        partition = 2
    elif test:
        partition = 3
    else:
        partition = 4
    plt.savefig('seq_{}_batch_{}_wi_{}_part_{}_rgb_{}.png'.format(
        seq_index,
        batch_index,
        window_index,
        partition,
        rgb))
    plt.close()


