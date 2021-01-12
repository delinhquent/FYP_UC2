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

def automate_experiment_name(tfidf, feature_select, normalize, valid_models):
    experiment_name = valid_models[str(model.lower())] + " - "

    feature_name = "All Features "
    if feature_select in ['y','Y']:
        feature_name = 'Important Features '

    tfidf_name = "without TFIDF"
    if tfidf in ['y','Y']:
        tfidf_name = "with TFIDF"

    if normalize in ['y','Y']:
        return experiment_name + feature_name + 'Normalized ' + tfidf_name
    
    return experiment_name + feature_name + tfidf_name


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
            model = args.model
            
            valid_models = {'dbscan': "DBSCAN",'isolation_forest': "Isolation Forest",'eif':"EIF"}

            if model == None:
                print("Please parse the model's names using -model respectively...")
            elif model == None:
                print("Please parse the model's names using -model_n respectively...")
            elif str(model).lower() not in valid_models.keys():
                print("Unable to find such models...")
            else:
                tfidf = str(args.tfidf).lower()
                feature_select = str(args.feature_select).lower()
                normalize = str(args.normalize).lower()

                valid_values = ['y','n','Y','N']

                if tfidf in valid_values and feature_select in valid_values and normalize in valid_values:
                    model_config = process_config(args.model_config)
                    comet_config = process_config(args.comet_config)

                    # Automatically create experiment name
                    experiment_name = automate_experiment_name(tfidf,feature_select,normalize, valid_models)

                    print("Logging experiment name: {name}".format(name=experiment_name))
                    experiment = Experiment(
                        api_key=comet_config.experiment.api_key,
                        project_name=comet_config.experiment.project_name,
                        workspace=comet_config.experiment.workspace
                    )
                    
                    experiment.set_name(experiment_name)

                    configs = [config, comet_config, model_config]
                    train_model(configs, experiment, model, tfidf)
                else:
                    print("Please input y or n for -tfidf/-feature_select/-normalize...")                  
        else:
            print("There is no such mode yet...")
            exit(0)
    except ValueError:
        print("Missing or invalid arguments")
        exit(0)





