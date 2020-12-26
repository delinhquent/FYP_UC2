from data_loader.data_loader import DataLoader
from preprocess.preprocessor import Preprocessor


class Generator:
    def __init__(self, config):
        self.config = config
        self.profiles_data_loader = DataLoader(config.profiles.base_data_path)
        self.review_activity_data_loader = DataLoader(config.review_activity.base_data_path)
        self.profiles_data = None
        self.review_activity_data = None
        self.preprocessor = None
        self.extractor = None

    def load_preprocessor(self):
        self.profiles_data = self.get_profiles_data()
        self.review_activity_data = self.get_review_activity_data()
        self.preprocessor = Preprocessor(self.config, self.profiles_data)

    def preprocess_review_activity(self):
        self.review_activity_data = self.preprocessor.preprocess_review_activity()

    def get_profiles_data(self):
        self.profiles_data_loader.load_data()
        return self.profiles_data_loader.get_data()
    
    def get_review_activity_data(self):
        self.review_activity_data_loader.load_data()
        return self.review_activity_data_loader.get_data()

    def save_review_activity_data(self):
        self.review_activity_data.to_csv(self.config.review_activity.base_data_path, index=False)