from data_loader.data_loader import DataLoader
from engineer.engineer import FeatureEngineer


class Engineer:
    def __init__(self, config):
        self.config = config
        self.reviews_data_loader = DataLoader(config.reviews.base_data_path)
        self.profiles_data_loader = DataLoader(config.profiles.base_data_path)
        self.products_data_loader = DataLoader(config.products.base_data_path)
        self.review_activity_data_loader = DataLoader(config.review_activity.base_data_path)
        self.reviews_data = None
        self.profiles_data = None
        self.products_data = None
        self.review_activity_data = None
        self.engineer = None

    def load_data(self):
        self.reviews_data = self.get_reviews_data()
        self.profiles_data = self.get_profiles_data()
        self.products_data = self.get_products_data()
        self.review_activity_data = self.get_review_activity_data()

    def load_engineer(self):
        self.engineer = FeatureEngineer(config = self.config, reviews_df = self.reviews_data, profiles_df = self.profiles_data, products_df = self.products_data, review_activity_df = self.review_activity_data)

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

    def engineer_reviews(self):
        self.reviews_data = self.engineer.engineer_reviews()

    def save_reviews_data(self):
        self.reviews_data.to_csv(self.config.reviews.save_data_path, index=False)
    
    def save_profiles_data(self):
        self.profiles_data.to_csv(self.config.profiles.save_data_path, index=False)
    
    def save_products_data(self):
        self.products_data.to_csv(self.config.products.save_data_path, index=False)
    
    def save_review_activity_data(self):
        self.review_activity_data.to_csv(self.config.review_activity.save_data_path, index=False)