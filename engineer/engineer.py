from utils.engineer import *

class FeatureEngineer:
    def __init__(self, config, reviews_df, profiles_df, products_df, review_activity_df):
        self.config = config
        self.reviews_df = reviews_df
        self.profiles_df = profiles_df
        self.products_df = products_df
        self.review_activity_df = review_activity_df

    def engineer_reviews(self):
        self.reviews_df = engineer_reviews(self.reviews_df,
                                              self.config.preprocessing.contractions_path,
                                              self.config.preprocessing.slangs_path)
        return self.reviews_df
