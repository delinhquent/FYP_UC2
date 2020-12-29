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
    
    engineer_all(data_engineer)
    
    print('Feature Engineering Pipeline Completed...')

def engineer_all(data_engineer):
    engineer_reviews_only(data_engineer)
    engineer_review_activity_only(data_engineer)
    engineer_profiles_only(data_engineer)
    engineer_products_only(data_engineer)

def engineer_reviews_only(data_engineer):
    print("Engineering features for reviews dataset...")
    data_engineer.engineer_reviews()
    
    print("Saving generated features into a dataset...")
    data_engineer.save_reviews_data()

def engineer_review_activity_only(data_engineer):
    print("Engineering features for review activity dataset...")
    data_engineer.engineer_review_activity()
    
    print("Saving generated features into a dataset...")
    data_engineer.save_review_activity_data()

def engineer_profiles_only(data_engineer):
    print("Engineering features for profiles dataset...")
    data_engineer.engineer_profiles()
    
    print("Saving generated features into a dataset...")
    data_engineer.save_profiles_data()

def engineer_products_only(data_engineer):
    print("Engineering features for products dataset...")
    data_engineer.engineer_products()
    
    print("Saving generated features into a dataset...")
    data_engineer.save_products_data()
