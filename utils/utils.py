import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='configs/config.json',
        help='The Configuration file')
    argparser.add_argument(
        '-d', '--dataset',
        default='all',
        help='Which dataset to generate [all, reviews, profiles, products, review_activity]')

    args = argparser.parse_args()
    return args
