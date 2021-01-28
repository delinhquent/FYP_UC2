from data_loader.data_loader import DataLoader

import pandas as pd

from preprocess.preprocessor import Preprocessor

from gensim.models import KeyedVectors
from gensim.models.fasttext import FastText
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile

import pickle

import os

from sklearn.feature_extraction.text import TfidfVectorizer

from utils.engineer_functions import temp_new_text
from utils.embedding import create_embedding_df


class Generator:
    def __init__(self, config):
        self.config = config
        self.profiles_data_loader = DataLoader(config.profiles.base_data_path)
        self.reviews_data_loader = DataLoader(config.reviews.base_data_path)
        self.profiles_data = None
        self.reviews_data = None
        self.review_activity_data = None
        self.preprocessor = None
        self.extractor = None
        self.tfidf_data = None
        self.glove_data = None
        self.fasttext_data = None
        self.word2vec_data = None

    def load_preprocessor(self):
        self.profiles_data = self.get_profiles_data()
        self.preprocessor = Preprocessor(self.config, self.profiles_data)

    def embed_words(self):
        self.reviews_data = self.get_reviews_data()

        self.tfidf_data = self.get_tfidf_vector()
        print("Saving embeddings into csv...")
        self.tfidf_data.to_csv(self.config.tfidf.reviews_vector, index= False)

        params = {
            "embedding_size": 300,
            "window_size": 5,
            "min_word": 5,
            "down_sampling": 1e-2
            }
        # self.fasttext_data = self.get_fasttext_vector(params)
        # print("Saving embeddings into csv...")
        # self.fasttext_data.to_csv(self.config.fasttext.reviews_vector, index= False)

        # self.word2vec_data = self.get_word2vec_vector(params)
        # print("Saving embeddings into csv...")
        # self.word2vec_data.to_csv(self.config.word2vec.reviews_vector, index= False)

        # self.glove_data = self.get_glove_vector()
        # print("Saving embeddings into csv...")
        # self.glove_data.to_csv(self.config.glove.reviews_vector, index= False)

    def preprocess_review_activity(self):
        self.review_activity_data = self.preprocessor.preprocess_review_activity()

    def get_tfidf_vector(self):
        print("Generating TFIDF Vector...")
        try:
            vec = TfidfVectorizer (ngram_range = (1,2),min_df=0.1,max_df=0.9)
            tfidf_reviews = vec.fit_transform(self.reviews_data['cleaned_text'].astype(str))

            pickle.dump(vec, open(self.config.tfidf.model_file,"wb"))
            # vec = pickle.load(open(self.config.tfidf.model_file, "rb")) # keeping this code for future development

            tfidf_reviews_df = pd.DataFrame(tfidf_reviews.toarray(), columns=vec.get_feature_names())
        except Exception as e:
            print(e)
        return tfidf_reviews_df.fillna(value=0)

    def get_fasttext_vector(self,params):
        # retrieve fasttext vector
        print("Generating fastText model...")
        ft_model = FastText(self.reviews_data['cleaned_text'].astype(str),
                    size=params["embedding_size"],
                    window=params["window_size"],
                    min_count=params["min_word"],
                    sample=params["down_sampling"],
                    sg=0, # put 1 if you want to use skip-gram, look into the documentation for other variables
                    iter=100)
        ft_model.save(self.config.fasttext.model_file)
        # ft_model = FastText.load(get_tmpfile(self.config.fasttext.model_file)) # keep this code for future development

        print("Generating Vector with fastText...")
        # ft_reviews_df = create_embedding_df(reviews, ft_model)
        num_features = ft_model.vector_size
        vocab_list = list(ft_model.wv.vocab)

        ft_reviews_df = self.reviews_data['cleaned_text'].astype(str).str.split().apply(avg_sentence_vector,model=ft_model,num_features = num_features,vocab = vocab_list)
        return ft_reviews_df.fillna(value=0)
    
    def get_word2vec_vector(self,params):
        # retrieve glove vector
        print("Generating Word2Vec model...")
        word2vec_model = Word2Vec(self.reviews_data['cleaned_text'].astype(str),
                    size=params["embedding_size"],
                    window=params["window_size"],
                    min_count=params["min_word"],
                    sample=params["down_sampling"],
                    sg=0, # put 1 if you want to use skip-gram, look into the documentation for other variables
                    iter=100)
        word2vec_model.save(self.config.word2vec.model_file)
        
        # word2vec_model = FastText.load(get_tmpfile(self.config.word2vec.model_file)) # keeping this code for future references

        print("Generating Vector with Word2Vec...")
        num_features = word2vec_model.vector_size
        vocab_list = list(word2vec_model.wv.vocab)

        word2vec_reviews_df = self.reviews_data['cleaned_text'].astype(str).str.split().apply(avg_sentence_vector,model=word2vec_model,num_features = num_features,vocab = vocab_list)
                    
        return word2vec_reviews_df.fillna(value=0)
    
    def get_glove_vector(self):
        if not os.path.exists(self.config.glove.word2vec_output_file):
            # retrieve glove vector
            print("Converting .txt file into .word2vec...")
            glove2word2vec(self.config.glove.glove_input_file, self.config.glove.word2vec_output_file)
        
        # load the Stanford GloVe model
        print("Loading GloVe model...")
        glove_model = KeyedVectors.load_word2vec_format(self.config.glove.word2vec_output_file, binary=False)
        
        print("Generating Vector with GloVe...")
        num_features = glove_model.vector_size
        vocab_list = list(glove_model.wv.vocab)

        word2vec_reviews_df = self.reviews_data['cleaned_text'].astype(str).str.split().apply(avg_sentence_vector,model=glove_model,num_features = num_features,vocab = vocab_list)
        
        return word2vec_reviews_df.fillna(value=0)

    def get_profiles_data(self):
        self.profiles_data_loader.load_data()
        return self.profiles_data_loader.get_data()
    
    def get_reviews_data(self):
        self.reviews_data_loader.load_data()
        return self.reviews_data_loader.get_data()
    
    def get_review_activity_data(self):
        self.review_activity_data_loader.load_data()
        return self.review_activity_data_loader.get_data()

    def save_review_activity_data(self):
        self.review_activity_data.to_csv(self.config.review_activity.base_data_path, index=False)
