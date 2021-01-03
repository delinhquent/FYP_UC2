# Usecase 2 - Fake Reviews Detector

[![Maintainability](https://api.codeclimate.com/v1/badges/e05270ae8c4746b5227c/maintainability)](https://codeclimate.com/github/delinhquent/FYP_UC2/maintainability)

[Final Year Project] Develop models to sift out reviews that seem unnatural, and then use a scoring algorithm to assign each review an impact score.

## Project Organization
------------

    ├── README.md                                   <- A simple walkthrough on how to use this application
    ├── data/
    │   ├── base/                                   <- Data is retrieved via DVC. 
    │   |   ├─ consolidated_profiles.csv            <- Profiles Dataset
    │   |   ├─ consolidated_products.csv            <- Reviews Dataset
    │   |   └─ consolidated_product_info.csv        <- Products Dataset
    |   |
    │   ├── preprocessing/                          <- Files used for doing preprocessing text.
    │   |   ├── contractions.txt                
    │   |   └── slangs.txt
    |   |
│   ├── raw/                                         <- Data is retrieved via DVC. 
    │   |   ├─ consolidated_profiles.csv
    │   |   ├─ consolidated_products.csv
    │   |   └─ consolidated_product_info.csv
    |   |
    │   └── uc2/                                     <- Workspace folder in remote server for this application.
    │       ├─ external/                             <- Data from third party sources or user inputs.
    │       |   ├─ consolidated_profiles.csv
    │       |   ├─ consolidated_products.csv
    │       |   └─ consolidated_product_info.csv
    │       |
    │       ├─ interim/                              <- Intermediate data that has been transformed.
    │       |   ├─ consolidated_review_activity.csv  <- Review Activity Dataset
    │       |   ├─ consolidated_products_feature_engineering.csv 
    │       |   ├─ consolidated_review_activity_feature_engineering.csv
    │       |   ├─ consolidated_profiles_feature_engineering.csv
    │       |   └─ consolidated_product_info_feature_engineering.csv
    │       |
    │       └─ processed/                            <- The final, canonical data sets for modeling. 
    │           ├─ fake_framework_features.csv       <- Modelling Dataset using Fake Features Framework
    │           └─ reviews_tfidf.csv                 <- Top 100 Features for TFIDF Vectors from Reviews
    │
    ├── models                                       <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                                    <- Jupyter notebooks for workings and visualizations.
    │
    ├── requirements.txt                             <- The requirements file for reproducing the analysis 
    |                                                environment, e.g. generated with `pip freeze > requirements.txt`
    │
    ├── src                                          <- Source code for use in this project.
    │   │
    │   ├── data                                     <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features                                 <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   └── models                                   <- Scripts to train models and then use trained 
    │       │                                            models to make predictions
    │       ├── predict_model.py
    │       └── train_model.py
    │
    ├── pipeline/                                    <- Scripts to load all methods and classes required 
    │   ├── generator.py   
    │   └── engineer.py
    |
    ├── preprocess/                                  <- Preprocessor class for preprocessing and cleaning steps
    |   └── preprocessor.py                                            
    |
    ├── engineer/                                    <- Engineer class for feature engineering 
    │   └── engineer.py
    |
    └── utils/                                       <- Functions for each classes 
--------

## Installation

Simply run the following after using a virtual environment.
```
pip install -r requirements.txt
```