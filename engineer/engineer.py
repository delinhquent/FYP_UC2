from utils.engineer import *

class FeatureEngineer:
    def __init__(self, **kwargs):
        valid_keys = ["config", "reviews_df", "profiles_df", "products_df", "review_activity_df", "loreal_brand_list","sample_incentivized_list"]
        for key in valid_keys:
            setattr(self, key, kwargs.get(key))

    def engineer_reviews(self):
        self.reviews_df = engineer_reviews(self.reviews_df, self.sample_incentivized_list)
        return self.reviews_df
    
    def engineer_review_activity(self):
        self.review_activity_df = engineer_review_activity(self.review_activity_df, self.loreal_brand_list, self.sample_incentivized_list)
        return self.review_activity_df
