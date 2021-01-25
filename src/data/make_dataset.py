"""
This script loads the required files, preprocesses it, and saves in a directory.
"""
from utils.config import process_config
from utils.utils import get_args
from pipeline.generator import Generator

def generate(config):
    # capture the config path from the run arguments
    # then process the json configuration file
    data_generator = Generator(config)

    print("Loading data and preprocessor...")
    data_generator.load_preprocessor()

    print("Extracting and preprocessing reviewer contributions into a dataset...")
    data_generator.preprocess_review_activity()
    
    print("Saving reviewer contributions into a dataset...")
    data_generator.save_review_activity_data()
    
    print("Embedding words into TFIDF, fastText, Word2Vec & GloVe...")
    data_generator.embed_words()

    print('Generated datasets are saved...')