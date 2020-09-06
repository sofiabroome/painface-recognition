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
                        help="Path to folder containing all data")
    parser.add_argument('--rgb-path', nargs='?', type=str,
                        help="Name of folder with all the rgb data to be processed")
    parser.add_argument('--of-path', nargs='?', type=str,
                        help="Name of folder with all the optical flow data to be processed")
    parser.add_argument('--config-file', nargs='?', type=str,
                        help="path to config file")
    parser.add_argument('--train-subjects', nargs='?', type=str,
                        help="List of subject-id:s to train on, choosing from range(0,6): ex [0,1,2,3]")
    parser.add_argument('--val-subjects', nargs='?', type=str,
                        help="List of subject-id:s to validate on, choosing from range(0,6): ex [4,5]")
    parser.add_argument('--test-subjects', nargs='?', type=str,
                        help="List of subject-id:s to test on, choosing from range(0,6): ex [4,5]")
    parser.add_argument('--subjects-overview', nargs='?', type=str,
                        help="List with ID:s of subjects in dataset.")
    parser.add_argument('--job-identifier', nargs='?', type=str,
                        help='Choose some string to identify the image of the training process')
    parser.add_argument('--test-run', nargs='?', type=int,
                        help='Whether to run as a quick test or not.')
    return parser.parse_args()


def read_default_args():
    parser = argparse.ArgumentParser()
    config_file = open("metadata/default_args.txt","r")
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
            elif key == '--subjects-overview':
                parser.add_argument(key, nargs='?', default=value, type=str, help=help)
            elif key == '--round-to-batch':
                parser.add_argument(key, nargs='?', default=value, type=bool, help=help)
            else:
                parser.add_argument(key, nargs='?', default=value, type=int, help=help)
    return parser.parse_args()
