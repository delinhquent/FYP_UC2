"""
This script loads the required files, preprocesses it, and saves in a directory.
"""
from comet_ml import Experiment

from utils.config import process_config
from utils.utils import get_args
from src.data.make_dataset import generate
from src.features.build_features import build_features
from src.models.train_model import train_model
import utils.common_resources

if __name__ == '__main__':
    try:
        args = get_args()
        config = process_config(args.config)
        mode = args.mode

        if mode == 'all':
            generate(config)
            build_features(config)
        if mode == 'dataset':
            generate(config)
        elif mode == 'build_features':
            build_features(config)
        elif mode == 'train_model':
            experiment_name = args.experiment_name
            model_name = args.model_name

            if experiment_name == None and model_name == None:
                print("Please parse the experiment's and model's names using -exp_n and -model_n respectively...")
            elif experiment_name == None:
                print("Please parse the experiment's name using -exp_n...")
            elif model_name == None:
                print("Please parse the model's names using -model_n respectively...")
            else:
                tfidf = str(args.tfidf).lower()
                if tfidf in ['y','n','Y','N']:
                    model_config = process_config(args.model_config)
                    comet_config = process_config(args.comet_config)

                    print("Logging experiment name: {name}".format(name=experiment_name))
                    experiment = Experiment(
                        api_key=comet_config.experiment.api_key,
                        project_name=comet_config.experiment.project_name,
                        workspace=comet_config.experiment.workspace
                    )
                    experiment.set_name(experiment_name)

                    configs = [config, comet_config, model_config]
                    train_model(configs, experiment, model_name, tfidf)
                else:
                    print("Please input y or n for -tfidf...")                  
        else:
            print("There is no such mode yet...")
            exit(0)
    except ValueError:
        print("Missing or invalid arguments")
        exit(0)




