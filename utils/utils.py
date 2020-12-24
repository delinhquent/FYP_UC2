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
        '-m', '--mode',
        default='',
        help='dataset - Generate Review Activity from reviewer contribution column from Profile Dataset')


    args = argparser.parse_args()
    return args
