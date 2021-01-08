import argparse
from argparse import RawTextHelpFormatter


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)

    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='configs/config.json',
        help='The Configuration file')
    argparser.add_argument(
        '-mc', '--model_config',
        dest='model_config',
        default='configs/model_config.json',
        help='The Configuration file for models')
    argparser.add_argument(
        '-cc', '--comet_config',
        dest='comet_config',
        default='configs/comet_config.json',
        help='The Configuration file for comet')
    argparser.add_argument(
        '-tfidf', '--tfidf',
        default= '',
        help='Train model with TFIDF. Default is y. Set n if you do not want to train model with TFIDF')
    argparser.add_argument(
        '-m', '--mode',
        default='',
        help="1. dataset - Generate Review Activity from reviewer contribution column from Profile Dataset.\n2. build_features - Generate dataset for modelling with feature engineering pipeline.\n3. train_model - Train Model with modelling dataset.\n10. all - Does all the modes mentioned above.")
    argparser.add_argument(
        '-exp_n', '--experiment_name',
        help='The name for the experiment. To be used with -m train_model.')
    argparser.add_argument(
        '-model_n', '--model_name',
        help='The name for the model. To be used with -m train_model.')


    args = argparser.parse_args()
    return args
