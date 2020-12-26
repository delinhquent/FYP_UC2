"""
This script loads the required files, preprocesses it, and saves in a directory.
"""
from utils.config import process_config
from utils.utils import get_args
from pipeline.engineer import Engineer

def build_features(config):
    # capture the config path from the run arguments
    # then process the json configuration file
    data_engineer = Engineer(config)

    print("Loading data...")
    data_engineer.load_data()

    print("Loading Engineer...")
    data_engineer.load_engineer()

    print("Engineering features...")
    data_engineer.engineer_reviews()
    
    print("Saving generated features into a dataset...")
    data_engineer.save_reviews_data()
    
    print('Generated datasets with features are saved...')