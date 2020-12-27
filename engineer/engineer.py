from utils.engineer import *

class FeatureEngineer:
    def __init__(self, **kwargs):
        valid_keys = ["config", "reviews_df", "profiles_df", "products_df", "review_activity_df"]
        for key in valid_keys:
            setattr(self, key, kwargs.get(key))

    def engineer_reviews(self):
        self.reviews_df = engineer_reviews(self.reviews_df,
                                              self.config.preprocessing.contractions_path,
                                              self.config.preprocessing.slangs_path)
        return self.reviews_df
    
    def engineer_review_activity(self):
        self.review_activity_df = engineer_review_activity(self.review_activity_df,
                                              self.config.preprocessing.contractions_path,
                                              self.config.preprocessing.slangs_path)
        return self.review_activity_df
