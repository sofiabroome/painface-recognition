import argparse

args_length = -1


class ArgParser:
    def __init__(self, args_length):
        self.args_length = args_length

    def parse(self):
        if self.args_length > 1:
            args = parse_arguments()
        else:
            args = read_default_args()
        return args


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', nargs='?', type=str,
                        help="Name of folder with all the data to be processed")
    parser.add_argument('--model', nargs='?', type=str,
                        help="Choose neural network architecture.")
    parser.add_argument('--input-width', nargs='?', type=int,
                        help="Input image width")
    parser.add_argument('--input-height', nargs='?', type=int,
                        help="Input image height")
    parser.add_argument('--nb-labels', nargs='?', type=int,
                        help="Number of unique labels for this dataset")
    parser.add_argument('--nb-lstm-layers', nargs='?', type=int,
                        help="Number of stacked LSTM layers")
    parser.add_argument('--nb-lstm-units', nargs='?', type=int,
                        help="Number of LSTM units")
    parser.add_argument('--nb-conv-filters', nargs='?', type=int,
                        help="Number of convolutional filters")
    parser.add_argument('--nb-dense-units', nargs='?', type=int,
                        help="Number of dense/fully connected units")
    parser.add_argument('--kernel-size', nargs='?', type=int,
                        help="Kernel size of convolutional filter")
    parser.add_argument('--dropout-1', nargs='?', type=float,
                        help="Probability of dropout 1")
    parser.add_argument('--dropout-2', nargs='?', type=float,
                        help="Probability of dropout 2")
    parser.add_argument('--nb-epochs', nargs='?', type=int,
                        help="Number of training epochs")
    parser.add_argument('--early-stopping', nargs='?', type=int,
                        help="Early stopping patience")
    parser.add_argument('--optimizer', nargs='?', type=str,
                        help="Choice of optimizer (can choose from Keras' different default optimizers)")
    parser.add_argument('--lr', nargs='?', type=float,
                        help="Learning rate")
    parser.add_argument('--batch-size', nargs='?', type=int,
                        help="Batch size")
    parser.add_argument('--round-to-batch', nargs='?', type=bool,
                        help='Choose whether to round the last batch to the specified batch size')
    parser.add_argument('--train-horses', nargs='?', type=str,
                        help="List of horse-id:s to train on, choosing from range(0,6): ex [0,1,2,3]")
    parser.add_argument('--val-horses', nargs='?', type=str,
                        help="List of horse-id:s to validate on, choosing from range(0,6): ex [4,5]")
    parser.add_argument('--test-horses', nargs='?', type=str,
                        help="List of horse-id:s to test on, choosing from range(0,6): ex [4,5]")
    parser.add_argument('--device', nargs='?', type=str,
                        help="Name of device to run on")
    parser.add_argument('--image-identifier', nargs='?', type=str,
                        help='Choose some string to identify the image of the training process')
    parser.add_argument('--test-run', nargs='?', type=int,
                        help='Whether to run as a quick test or not.')
    parser.add_argument('--seq-length', nargs='?', type=int,
                        help="Length of sequence for LSTM and for 5D input stuff")
    parser.add_argument('--nb-workers', nargs='?', type=int,
                        help="Number of workers")
    parser.add_argument('--nb-input-dims', nargs='?', type=int,
                        help="Number of input dimensions")
    parser.add_argument('--val-fraction', nargs='?', type=int,
                        help="Whether to use val fract instead of separate horses. 0 false 1 true.")
    parser.add_argument('--data-type', nargs='?', type=str,
                        help='The type of input data')
    return parser.parse_args()


def read_default_args():
    parser = argparse.ArgumentParser()
    config_file = open("default_args.txt","r")
    for line in config_file.readlines():
        key_value_help = line.split(",")
        try:
            key = key_value_help[0]
            value = key_value_help[1]
            help = key_value_help[2]
        except Exception as e:
            print("Invalid format of config file, ignoring...")
        else:
            if key == '--file-name' or key == '--devices':
                parser.add_argument(key, nargs='?', default=value, type=str, help=help)
            elif key == '--round-to-batch':
                parser.add_argument(key, nargs='?', default=value, type=bool, help=help)
            else:
                parser.add_argument(key, nargs='?', default=value, type=int, help=help)
    return parser.parse_args()
