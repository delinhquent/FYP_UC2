# Fake Reviews Detector

[![Maintainability](https://api.codeclimate.com/v1/badges/e05270ae8c4746b5227c/maintainability)](https://codeclimate.com/github/delinhquent/FYP_UC2/maintainability)

The scope of the use case is to develop a model to sift out reviews that seem unnatural, and then use a scoring algorithm to assign each review an impact score. The anomaly detection will be developed using white box models such as Dbscan (Density Based Spatial Clustering of Applications with Noise) and Isolation Forests.This provides transparency and provides an understanding of the influencing variables, which ensures accountability to the end users.

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
    │   ├── raw/                                    <- Data is retrieved via DVC. 
    │   |   ├─ consolidated_profiles.csv
    │   |   ├─ consolidated_products.csv
    │   |   └─ consolidated_product_info.csv
    |   |
    │   └── uc2/                                     <- Workspace folder in remote server
    │       ├─ external/                             <- Data from third party sources/inputs.
    │       |   ├─ loreal_brand_list.csv
    │       |   └─ sample_incentivized_reviews.csv
    │       |
    │       ├─ interim/                              <- Intermediate data that has been transformed.
    │       |   └─ consolidated_review_activity.csv  <- Review Activity Dataset
    │       |
    │       └─ processed/                            <- The final, canonical data sets for modeling. 
    │           └─ fake_framework_features.csv       <- Modelling Dataset using Fake Features Framework
    │
    ├── models                                       <- Trained models, predictions & summaries
    │
    ├── notebooks                                    <- Jupyter notebooks for workings & visualizations.
    │
    ├── requirements.txt                             <- The requirements file for environment
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
    │
    ├── pipeline/                                    <- Scripts to load all methods & classes required 
    │   ├── generator.py   
    │   ├── engineer.py
    │   └── trainer.py
    |
    ├── preprocess/                                  <- Preprocessor class for preprocessing & cleaning
    |   └── preprocessor.py                                            
    |
    ├── engineer/                                    <- Engineer class for feature engineering 
    │   └── engineer.py
    |
    ├── trainers/                                    <- Trainer class for modelling 
    │   └── dbscan.py
    |
    └── utils/                                       <- Functions for each classes 
--------

## Installation

Simply run the following after using a virtual environment.
```
pip install -r requirements.txt
```