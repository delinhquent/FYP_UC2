"""
This script loads the required files, preprocesses it, and saves in a directory.
"""


from utils.config import process_config
from utils.utils import get_args

from pipeline.trainer import Trainer


def train_model(configs, experiment, model, tfidf):
    # capture the config path from the run arguments
    # then process the json configuration file    
    config = configs[0]
    comet_config = configs[1]
    model_config = configs[2]

    train_model = Trainer(config = config, comet_config = comet_config, model_config = model_config, experiment = experiment, model = model,tfidf = tfidf)

    print("Loading data...")
    train_model.load_data()

    print("Loading trainer...")
    if model == "dbscan":
        train_model.experiment_dbscan()
    else:
        print("Unable to find such models...")

    