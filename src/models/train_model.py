"""
This script loads the required files, preprocesses it, and saves in a directory.
"""


from utils.config import process_config
from utils.utils import get_args

from pipeline.trainer import Trainer


def train_model(config, comet_config, model_config, experiment, model_name, tfidf):
    # capture the config path from the run arguments
    # then process the json configuration file    
    dbscan_model = Trainer(config = config, comet_config = comet_config, model_config = model_config, experiment = experiment, model_name = model_name,tfidf = tfidf)

    print("Loading data...")
    dbscan_model.load_data()

    print("Loading trainer...")
    dbscan_model.experiment_dbscan()

    