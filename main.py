"""
This script loads the required files, preprocesses it, and saves in a directory.
"""
from utils.config import process_config
from utils.utils import get_args
from src.data.make_dataset import generate

if __name__ == '__main__':
    try:
        args = get_args()
        config = process_config(args.config)
        mode = args.mode

        if mode == 'dataset':
            generate(config)
        else:
            print("There is no such mode yet...")
            exit(0)
    except ValueError:
        print("Missing or invalid arguments")
        exit(0)



