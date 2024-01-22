from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.data_processor import load_json_data, load_csv_data, create_id_map, CSVData, JSONData
from scripts.utils import mean_pooling, get_logger

import pdb


class BaseDataSet(Dataset, ABC):
    logger = get_logger()

    def __init__(self, json_path, csv_path, **model_params):
        self.json_dict, self.csv_dict = self.load_data(json_path, csv_path)
        self.csv_data = CSVData(self.csv_dict)

        self.user_to_movie = {}
        ratings = self.csv_dict['ratings']
        for i in ratings['userId'].tolist():
            self.user_to_movie[i] = list(zip(ratings[ratings['userId'] == i]['movieId'].tolist(),
                                             ratings[ratings['userId'] == i]['rating'].tolist(),
                                             ratings[ratings['userId'] == i]['timestamp'].tolist()))
        movies = self.csv_dict['movies']
        self.id_to_movie = dict(zip(movies['movieId'], movies['title']))

        tmbd_to_movieid = create_id_map(self.csv_dict)
        self.json_data = JSONData(self.json_dict, id_map=tmbd_to_movieid)

        self.tokenizer = None
        self.model = None
        self.load_model(**model_params)

        self.features = self.generate_features()

    @staticmethod
    def load_data(json_path, csv_path):
        json_data = load_json_data(file_dir=json_path)
        csv_data = load_csv_data(file_dir=csv_path)
        return json_data, csv_data

    def generate_features(self):
        raise NotImplementedError

    def load_model(self, **kwargs):
        pass

    def __len__(self):
        return len(self)

    def __getitem__(self, idx):
        pass


class TMDBData(BaseDataSet, ABC):
    def load_model(self, **model_params):
        # load the encoder for embedding text features
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            self.tokenizer = AutoTokenizer.from_pretrained(model_params['tokenizer'])
            self.model = AutoModel.from_pretrained(model_params['model'])
            self.logger.info(f"loaded model {model_params['model']}")
        else:
            self.logger.info(f"GPU not available, using tf-idf instead.")

    def generate_features(self):
        movie_descr = self.json_data.process_context()
        movie_id_to_row = dict(zip(movie_descr['movieId'], movie_descr.index))
        row_to_movie_id = {movie_id_to_row[m]: m for m in movie_id_to_row}
        movie_genre = self.csv_data.process_movie_genre()
        movie_genre = movie_genre.loc[movie_genre['movieId'].isin(movie_descr['movieId'])]
        # then join the two feature matrices to get the final representation
        overview_tagline = [" ".join(p) for p in zip(movie_descr['overview'], movie_descr['tagline'])]
        if self.tokenizer is not None:
            input_text = self.tokenizer(overview_tagline,
                                        padding=True,
                                        truncation=True,
                                        return_tensors='pt')

            # Compute token embeddings
            with torch.no_grad():
                embeddings = self.model(**input_text)
            sentence_embeddings = mean_pooling(embeddings, input_text['attention_mask'])
            self.logger.info(f"embedded sentences dimension is {sentence_embeddings.shape()}")

            return sentence_embeddings

        self.logger.info(f"embedding with tf-idf feature.")
        vectorizer = TfidfVectorizer(
            max_df=0.5,
            min_df=5,
            stop_words="english",
        )
        movie_tfidf = vectorizer.fit_transform(overview_tagline)

        svd = TruncatedSVD(n_components=200, n_iter=7, random_state=42)
        content_svd = make_pipeline(svd, Normalizer(copy=False))
        sentence_embeddings = content_svd.fit_transform(movie_tfidf)

        return sentence_embeddings, movie_id_to_row, row_to_movie_id


class MovieData(BaseDataSet, ABC):
    def generate_features(self):
        movie_features_csv = self.csv_data.process_movie_csv_data()
        movie_features_json = self.json_data.process_features()
        movie_genre = self.csv_data.process_movie_genre()
        user_features_csv = self.csv_data.process_user_csv_data()

        # then join the movie features
        movie_feat = movie_features_json.merge(movie_genre, how='left', on='movieId')
        movie_feat = movie_feat.merge(movie_features_csv, how='left', on='movieId')

        movie_id_to_row = dict(zip(movie_feat['movieId'], movie_feat.index))

        # then for each user, find the movies in the user history and concat with corresponding movie feature
        # TODO: add ratings to the features!!
        full_feat_df = None
        for user in user_features_csv['userId']:
            user_ratings = dict([(m, [r, t]) for m, r, t in self.user_to_movie[user]])
            user_df = user_features_csv.loc[user_features_csv['userId'] == user]
            movie_ids = self.csv_data.user_movie_map[user]
            movie_df = movie_feat[movie_feat['movieId'].isin(movie_ids)].copy()

            movie_df['rating'] = np.nan
            for m_id in user_ratings:
                movie_df.loc[movie_df['movieId'] == m_id, 'rating'] = user_ratings[m_id][0]

            user_df = pd.concat([user_df] * movie_df.shape[0], ignore_index=True)
            user_block = pd.concat([user_df, movie_df.set_index(user_df.index)], axis=1)
            if full_feat_df is None:
                full_feat_df = user_block
            else:
                full_feat_df = pd.concat([full_feat_df, user_block], axis=0)

        return full_feat_df, movie_feat, user_features_csv
