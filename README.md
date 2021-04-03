# Unnatural Reviews Detector
<!-- [![Maintainability](https://api.codeclimate.com/v1/badges/e05270ae8c4746b5227c/maintainability)](https://codeclimate.com/github/delinhquent/FYP_UC2/maintainability) -->
![Placehoder for image](https://dl.dropboxusercontent.com/s/tv5wtmzpfug5u2p/fake_fact.jpg?dl=0)

## Description

The scope of the use case is to develop a model to sift out reviews that seem unnatural for L'Oréal based on their products. Anomaly Detection models such as One-Class Support Vector Machine and Isolation Forests are experimented. Evaluation of the models are done mainly using F1, Recall & Precision.

A scoring algorithm is then used to assign each review an impact score. The aim of such a scoring algorithm is to help users quickly prioritize which reviews they should investigate or take actions against based on its severity (could be in terms of money or brand).

## Read more about the project

[Click here to read more about the project](https://github.com/delinhquent/unnatural-reviews-detector/wiki)   

## Click to View Demo

[![This is a demo application for L'Oréal](https://img.youtube.com/vi/ZGmNutrkZAM/0.jpg)](https://www.youtube.com/watch?v=ZGmNutrkZAM)


## Project Organization
------------

    ├── README.md                                   <- A simple walkthrough on how to use this application
    │
    │
    ├── configs/                                      <- Configuration files
    │   ├── comet_config.json   
    │   ├── config.json
    │   └── model_config.json
    │
    │
    ├── data/
    │   ├── base/                                   <- Data is retrieved via DVC. 
    │   |   ├─ consolidated_profiles.csv            <- Profiles Dataset
    │   |   ├─ consolidated_products.csv            <- Reviews Dataset
    │   |   └─ consolidated_product_info.csv        <- Products Dataset
    |   |
    |   |
    │   ├── preprocessing/                          <- Files used for doing preprocessing text.
    │   |   ├── contractions.txt                
    │   |   └── slangs.txt
    |   |
    │   ├── raw/                                    <- Data is retrieved via DVC. 
    │   |   ├─ consolidated_profiles.csv
    │   |   ├─ consolidated_products.csv
    │   |   └─ consolidated_product_info.csv
    |   |
    │   └── uc2/                                     <- Workspace folder in remote server
    │       ├─ external/                             <- Data from third party sources/inputs.
    │       |   ├─ loreal_brand_list.csv
    │       |   ├─ sample_incentivized_reviews.csv
    │       |   └─ scrape_products.csv
    │       |
    │       ├─ interim/                              <- Intermediate data that has been transformed.
    │       |   ├─ consolidated_product_info.csv     <- Products dataset with new features engineered
    │       |   ├─ consolidated_products_labelled.csv  <- Review Dataset with labels (to train & evaluate model) 
    |       |   |                                           - labels are under `manual_label`
    │       |   ├─ consolidated_profiles.csv         <- Profiles dataset with new features engineered
    │       |   ├─ consolidated_review_activity.csv  <- Review Activity Dataset from `reviewer_contribution` column in Profiles Dataset
    │       |   └─ doc2vec_embedding.csv             <- Doc2Vec Embedded Values from reviews' text
    │       |
    │       └─ processed/                            <- The final, canonical data sets for modeling. 
    │           ├─ fake_framework_features.csv       <- Modelling Dataset using Fake Features Framework
    │           └─ suspicious_reviewers_metrics.csv  <- Profiles dataset with new features engineered and model's output
    │
    │
    ├── data_loader/                                <- Class to load data
    │   └─ data_loader.py   
    │
    │
    ├── engineer/                                   <- Class to conduct feature engineering
    │   └─ engineer.py   
    │
    │
    ├── featureselector/                            <- Class to select important features
    │   └─ featureselector.py  
    │
    │
    ├── images/                                     <- Folder to contain images for demo application
    │
    │
    ├── impactscorer/                                <- Class to score impact scores to business
    │   └─ impactscorer.py  
    │
    │
    ├── models                                       <- Trained models, predictions & summaries
    │   ├── results                                  <- Prediction results from models
    │   ├── word_embedding                           <- Word Embedding Model for Doc2Vec
    │   └── normalizer                               <- Pickled files for TFIDF & normalizing values
    │
    ├── notebooks                                    <- Jupyter notebooks for workings & visualizations.
    │
    │
    ├── pipeline/                                    <- Scripts to load all methods & classes required 
    │   ├── generator.py   
    │   ├── engineer.py
    │   └── trainer.py
    |
    |
    ├── preprocess/                                  <- Preprocessor class for preprocessing & cleaning
    |   └── preprocessor.py                                            
    |
    |
    ├── engineer/                                    <- Engineer class for feature engineering 
    │   └── engineer.py
    |
    |
    ├── src                                          <- Source code for use in this project.
    │   ├── data                                     <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features                                 <- Scripts to turn data into features for modeling
    │   │   └── build_features.py
    │   │
    │   └── models                                   <- Scripts to train models to make predictions
    │       ├── predict_model.py
    │       └── train_model.py
    |
    |
    ├── src                                          <- Source code for use in this project.
    │   ├── data                                     <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features                                 <- Scripts to turn data into features for modeling
    │   │   └── build_features.py
    │   │
    │   └── models                                   <- Scripts to train models
    │       └── train_model.py
    |
    |
    ├── trainers/                                    <- Trainer class for modelling 
    │   ├── dbscan.py
    │   ├── isolationforest.py
    │   ├── lof.py
    │   ├── lof_tuner.py
    │   ├── ocsvm_datashift_algorithm.py
    │   ├── ocsvm_tuner.py
    │   ├── pyodmodel.py
    │   └── rrcf.py
    |
    |
    ├── utils/                                       <- Functions for each classes 
    │   ├── clean.py
    │   ├── common_resources.py
    │   ├── config.py
    │   ├── em.py
    │   ├── em_bench_high.py
    │   ├── embedding.py
    │   ├── engineer.py
    │   ├── engineer_functions.py
    │   ├── extract.py
    │   ├── reformat.py
    │   ├── streamlit_functions.py
    │   └── utils.py
    |
    |
    ├── app.py                                      <- Demo Application for users to interact with
    |
    |
    ├── environment.yml                             <- yml environment for application
    |
    |
    ├── main.py                                      <- Main script file to run application 
    |
    |
    ├── requirements.txt                             <- The requirements file for environment
    |
    |
    └── SessionState.py                              <- Script to add per-session state for Streamlit's authentication

--------

## Installation (Either follow Step 1 or Step 2 before using the application)

### 1) Activate Virtual Environment from yml File
Create virtual environment from yml file:
```
conda env create -f environment.yml
```

Activate virtual environment.
```
conda activate usecase2
```

### 2) Pip Install Files after Creating a Virtual Environment
Simply run the following after using a virtual environment.
```
pip install -r requirements.txt
```


## Usage
### Run Application to Train Model via Data Version Control
To keep track of changes to the data in the remote server's working directory, Data Version Control (DVC) was used. This is similar to how GitHub tracks code changes, but DVC is meant for data.

1) Ensure DVC has been installed
```
pip install dvc
```
2) After installing dvc and initializing git, we will need to initialize dvc
```
# Initialize dvc
dvc init
# commit changes (.dvc folders)
git commit -m "Initialize DVC"
```

3) Track the folders which are of interest (in this case, its the `/data/raw` and `/data/base folders`)
```
dvc add data/raw data/base
git add data/raw.dvc data/base.dvc
git commit -m "added data folders"
```

4) Add the remote storage
```
dvc remote add -f -d remote ssh://<remote storage>
```

5) As the data pipeline has already been defined in the `dvc.yaml` file, only a single command is needed to execute the entire process. It will only run the necessary commands if they find any changes in the dependencies file.
```
dvc repro
```

### Run Script to Train Model via Anaconda Prompt
To run the script via Anaconda Prompt, the sequence, commands and file dependencies can be found in the `dvc.yaml` file.

Simply ensure that the file dependencies are present before executing the commands. Please also run the commands in sequence for your initial run as there are some file dependencies which are generated from the previous stage.


### Run Demo Application
```
streamlit run app.py
```
