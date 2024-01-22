import pdb
import random
import pickle
import torch
import pandas as pd

from scripts.datasets import MovieData, TMDBData
from scripts.models import ContentKNNModel, ContentAutoEncoder, UserMLPModel, UserXGBModel


class Ranker:
    # use KNN to find the most similar movies, then rank by predicted user rating and rules
    def __init__(self,
                 json_path,
                 csv_path,
                 content_model_path=None,
                 user_model_path=None,
                 top_k=10,
                 **kwargs):
        self.content_model_path = content_model_path
        self.user_model_path = user_model_path

        print(f"loading user-movie interaction features")
        self.user_features = MovieData(json_path, csv_path)  # joined mat, user mat, movie mat
        print(f"loading movie text features")
        self.content_features = TMDBData(json_path, csv_path, **kwargs)  # embeddings, movie id to row id, row to movie

        # if the model path is provided, use pretrained NN model for inference, otherwise use simpler models
        if content_model_path is not None:
            self.content_model = ContentAutoEncoder(**kwargs)
            self.content_model.load_state_dict(torch.load(content_model_path))
            self.content_model.eval()
        else:
            print(f"initializing KNN model for movie text features")
            self.content_model = ContentKNNModel()
            print(f"initializing KNN model training")
            self.content_model.fit(self.content_features.features[0])

        if user_model_path is not None:
            self.user_model = UserMLPModel(**kwargs)
            self.user_model.load_state_dict(torch.load(user_model_path))
            self.user_model.eval()
        else:
            print(f"initializing xgboost model for user-movie interaction features")
            self.user_model = UserXGBModel()
            print(f"initializing xgboost model training")
            self.user_model.fit(self.user_features.features[0], 'rating')

        self.top_k = top_k
        print(f"ranker has been successfully initialized!")

    def get_similar_movies(self, user_id, n_sample=10):
        # 1. use user_id to identify user movie history
        # 2. sample from user history (sample min(10, user_history))
        # 3. get k similar movies to each of the sampled user history movies
        sampled_movies = self.sample_movies(user_id, n_sample)
        embeddings, movie_to_row, row_to_movie = self.content_features.features
        embed_ids = [movie_to_row[m] for m, r, t in sampled_movies]  # return a list of movieIds
        sample_embeds = embeddings[embed_ids]
        results = self.content_model.inference(sample_embeds)
        res = {}
        for j, m in enumerate(sampled_movies):
            res[m[0]] = sorted(list(zip(*results[j])), key=lambda x: x[1], reverse=True)
        return res

    def get_user_rating_score(self, user_id, movie_ids):
        # user the user_model to predict the user-movie score
        joined_feat, movie_feat, user_feat = self.user_features.features
        movie_feat_df = movie_feat.loc[movie_feat['movieId'].isin(movie_ids)]
        movie_feat_df = movie_feat_df.reset_index()
        row_to_movie = dict(zip(movie_feat_df.index, movie_feat_df['movieId']))
        user_feat_df = user_feat.loc[user_feat['userId'] == user_id]
        user_feat_df = pd.concat([user_feat_df] * movie_feat_df.shape[0], ignore_index=True)
        joined_feat_df = pd.concat([user_feat_df, movie_feat_df.set_index(user_feat_df.index)], axis=1)
        # TODO: add step to remove movieId column (or should this be handled by the model?)
        x = joined_feat_df.loc[:, joined_feat_df.columns[~joined_feat_df.columns.isin(['userId', 'movieId', 'rating', 'level_0', 'index'])]]

        scores = self.user_model.inference(x)
        return scores, row_to_movie

    def sample_movies(self, user_id, n_sample):
        user_movies = sorted(self.user_features.user_to_movie[user_id], key=lambda x: (x[1], x[2]), reverse=True)
        user_movies = [user_movie for user_movie in user_movies if user_movie[0] in self.content_features.features[1]]
        n_sample = min(n_sample, len(user_movies))
        sampled_index = random.sample(range(len(user_movies)), n_sample)
        return [user_movies[j] for j in sampled_index]

    def make_recommendation(self, user_id, n_sample=5, rules=None):
        # include rules for recommendations
        if not rules:
            movie_ids = self.get_similar_movies(user_id, n_sample=n_sample)
            scores, row_to_movie = self.get_user_rating_score(user_id, movie_ids)
            score_to_id = sorted([(s, row_to_movie[i]) for i, s in enumerate(scores)], key=lambda x: x[0], reverse=True)
            return score_to_id[:self.top_k]

        else:
            # determine recommendations considering business rules that maximize rental return
            raise NotImplementedError
