# Unnatural Reviews Detector
------------

<!-- [![Maintainability](https://api.codeclimate.com/v1/badges/e05270ae8c4746b5227c/maintainability)](https://codeclimate.com/github/delinhquent/FYP_UC2/maintainability) -->

The scope of the use case is to develop a model to sift out reviews that seem unnatural for L'Oréal based on their products. A scoring algorithm is then used to assign each review an impact score. The anomaly detection are experimented with models such as One-Class Support Vector Machine and Isolation Forests. Evaluation of the models are done mainly using F1, Recall & Precision. We also included Excess Mass & Mass Volumes.

------------

## Demo Video
[![This is a demo application for L'Oréal]](https://youtu.be/ZGmNutrkZAM)

------------

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
    │       │
    │       ├─ external/                             <- Data from third party sources/inputs.
    │       |   ├─ loreal_brand_list.csv
    │       |   ├─ sample_incentivized_reviews.csv
    │       |   └─ scrape_products.csv
    │       |
    │       ├─ interim/                              <- Intermediate data that has been transformed.
    │       |   ├─ consolidated_product_info.csv     <- Products dataset with new features engineered
    │       |   ├─ consolidated_products_labelled.csv  <- Review Activity Dataset with labels (to train & evaluate model)
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
    │   │
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
    │   │
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

--------

## Usage

### Run Application to Train Model
```
dvc repro
```

### Run Demo Application
```
streamlit run app.py
```