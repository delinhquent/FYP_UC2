# from utils.reformat import reformat_profiles_df, reformat_reviews_df, reformat_products_df
from utils.extract import extract_review_activity


class Preprocessor:
    def __init__(self, config, review_activity_df):
        self.config = config
        self.review_activity_df = review_activity_df

    def preprocess_review_activity(self):
        self.review_activity_df = extract_review_activity(self.review_activity_df)
        # self.review_activity_df = reformat_reviews_df(self.review_activity_df,
        #                                       self.config.preprocessing.contractions_path,
        #                                       self.config.preprocessing.slangs_path)
        return self.review_activity_df
