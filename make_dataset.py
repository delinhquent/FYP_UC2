"""
This script loads the required files, preprocesses it, and saves in a directory.
"""
from utils.config import process_config
from utils.utils import get_args
from pipeline.generator import Generator

def generate():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except ValueError:
        print("Missing or invalid arguments")
        exit(0)
    dataset = args.dataset

    print('Creating the data generator...')
    data_generator = Generator(config)

    print('Loading the Data and Preprocessor...')

    data_generator.load_data()
    data_generator.load_preprocessor()

    print('Preprocessing data..')
    if dataset == 'all':
        print("In development...")
    elif dataset == 'reviews':
        print("In development...")
    elif dataset == 'profiles':
        print("In development...")
    elif dataset == 'products':
        print("In development...")
    elif dataset == 'review_activity':
        print("Extracting and preprocessing reviewer contributions into a dataset...")
        data_generator.preprocess_review_activity()
        
        print("Saving reviewer contributions into a dataset...")
        data_generator.save_review_activity_data()

    print('Generated datasets are saved...')

if __name__ == '__main__':
    generate()




