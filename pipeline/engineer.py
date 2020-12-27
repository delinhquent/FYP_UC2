from data_loader.data_loader import DataLoader
from engineer.engineer import FeatureEngineer

from utils.clean import clean_text


class Engineer:
    def __init__(self, config):
        self.config = config
        self.reviews_data_loader = DataLoader(config.reviews.base_data_path)
        self.profiles_data_loader = DataLoader(config.profiles.base_data_path)
        self.products_data_loader = DataLoader(config.products.base_data_path)
        self.review_activity_data_loader = DataLoader(config.review_activity.base_data_path)
        self.loreal_brand_data_loader = DataLoader(config.user_inputs.loreal_brand_path)
        self.sample_incentivized_data_loader = DataLoader(config.user_inputs.sample_incentivized_path)
        self.reviews_data = None
        self.profiles_data = None
        self.products_data = None
        self.review_activity_data = None
        self.loreal_brand_data = None
        self.sample_incentivized_data = None
        self.engineer = None

    def load_data(self):
        self.reviews_data = self.get_reviews_data()
        self.profiles_data = self.get_profiles_data()
        self.products_data = self.get_products_data()
        self.review_activity_data = self.get_review_activity_data()
        self.loreal_brand_data = self.get_loreal_brand_data()
        self.sample_incentivized_data = self.get_sample_incentivized_data()

    def load_engineer(self):
        self.engineer = FeatureEngineer(config = self.config, reviews_df = self.reviews_data, profiles_df = self.profiles_data, products_df = self.products_data, review_activity_df = self.review_activity_data, loreal_brand_list = self.loreal_brand_data, sample_incentivized_list = self.sample_incentivized_data)

    def get_reviews_data(self):
        self.reviews_data_loader.load_data()
        return self.reviews_data_loader.get_data()

    def get_profiles_data(self):
        self.profiles_data_loader.load_data()
        return self.profiles_data_loader.get_data()
    
    def get_products_data(self):
        self.products_data_loader.load_data()
        return self.products_data_loader.get_data()
    
    def get_review_activity_data(self):
        self.review_activity_data_loader.load_data()
        return self.review_activity_data_loader.get_data()

    def get_loreal_brand_data(self):
        self.loreal_brand_data_loader.load_data()
        temp_loreal_brand_data = list(self.loreal_brand_data_loader.get_data()['text'].str.replace('\n', ' ').str.replace('\t', ' ').str.lower().str.strip())
        return list(set(temp_loreal_brand_data + clean_text(temp_loreal_brand_data, self.config.preprocessing.contractions_path, self.config.preprocessing.slangs_path)))

    def get_sample_incentivized_data(self):
        self.sample_incentivized_data_loader.load_data()
        temp_sample_incentivized_data = list(self.sample_incentivized_data_loader.get_data()['text'].str.replace('\n', ' ').str.replace('\t', ' ').str.lower().str.strip())
        return list(set(temp_sample_incentivized_data + clean_text(temp_sample_incentivized_data, self.config.preprocessing.contractions_path, self.config.preprocessing.slangs_path)))

    def engineer_reviews(self):
        self.reviews_data = self.engineer.engineer_reviews()

    def engineer_review_activity(self):
        self.review_activity_data = self.engineer.engineer_review_activity()

    def save_reviews_data(self):
        self.reviews_data.to_csv(self.config.reviews.save_data_path, index=False)
    
    def save_profiles_data(self):
        self.profiles_data.to_csv(self.config.profiles.save_data_path, index=False)
    
    def save_products_data(self):
        self.products_data.to_csv(self.config.products.save_data_path, index=False)
    
    def save_review_activity_data(self):
        self.review_activity_data.to_csv(self.config.review_activity.save_data_path, index=False)