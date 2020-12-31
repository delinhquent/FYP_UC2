from utils.engineer import *

class FeatureEngineer:
    def __init__(self, **kwargs):
        valid_keys = ["config", "reviews_df", "profiles_df", "products_df", "review_activity_df", "loreal_brand_list","sample_incentivized_list"]
        for key in valid_keys:
            setattr(self, key, kwargs.get(key))

    def engineer_reviews(self):
        self.reviews_df = engineer_reviews(self.reviews_df, self.sample_incentivized_list, self.products_df, self.config.tfidf_vector.save_data_path)
        return self.reviews_df
    
    def engineer_review_activity(self):
        self.review_activity_df = engineer_review_activity(self.review_activity_df, self.loreal_brand_list, self.sample_incentivized_list)
        return self.review_activity_df
    
    def engineer_profiles(self):
        self.profiles_df = engineer_profiles(self.profiles_df, self.review_activity_df)
        return self.profiles_df

    def engineer_products(self):
        self.products_df = engineer_products(self.products_df, self.profiles_df, self.reviews_df)
        return self.products_df
