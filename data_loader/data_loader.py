import pandas as pd


class DataLoader:
    def __init__(self, path):
        self.path = path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.path, encoding='utf-8')

    def get_data(self):
        return self.df
