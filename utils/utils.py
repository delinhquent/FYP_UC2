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
        '-m', '--mode',
        help="1. dataset - Generate Review Activity from reviewer contribution column from Profile Dataset.\n2. build_features - Generate dataset for modelling with feature engineering pipeline.\n3. train_model - Train Model with modelling dataset.\n10. all - Does all the modes mentioned above.")
    argparser.add_argument(
        '-model', '--model',
        help="The model which you will be using. To be used with -m train_model.\nModel Name | Parse Value:\n1. DBSCAN | dbscan \n2. Isolation Forest | isolation_forest\n3. Extended Isolation Forest | eif")
    argparser.add_argument(
        '-tfidf', '--tfidf',
        help='Train model with TFIDF. Accepted values are "y" and "n".')
    argparser.add_argument(
        '-feature_select', '--feature_select',
        help='Train model with Feature Selection. Accepted values are "y" and "n".')
    argparser.add_argument(
        '-normalize', '--normalize',
        help='Train model with Normalization. Accepted values are "y" and "n".')

    args = argparser.parse_args()
    return args
