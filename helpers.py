def get_last_characters_from_string(string, nb_chars):
    return string[-nb_chars:]


def get_string_after_last_occurence_of_certain_char(string, char):
    left, right = string.rsplit(sep=char, maxsplit=1)
    return right
