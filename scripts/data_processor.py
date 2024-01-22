import os
import json
from glob import glob
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import pandas as pd


def load_json_data(file_dir):
    total_dict = {}
    json_files = glob(os.path.join(file_dir, "*.json"))
    for jfile in json_files:
        with open(jfile, "r") as jf:
            try:
                jdict = json.load(jf)
                key = os.path.splitext(os.path.basename(jfile))[0]
                total_dict[key] = jdict
            except Exception as e:
                print(f"Exception occurred when loading file {jfile} with message {e}.")
    return total_dict


def load_csv_data(file_dir):
    csv_dict = {}
    csv_files = glob(os.path.join(file_dir, "*.csv"))
    for csv_file in csv_files:
        data_df = pd.read_csv(csv_file, dtype={'movieId': object})
        key = os.path.splitext(os.path.basename(csv_file))[0]
        csv_dict[key] = data_df
    return csv_dict


def create_id_map(csv_dict):
    id_map_df = csv_dict['links'].dropna()
    tmbd_to_movieid = dict(zip(id_map_df['tmdbId'].astype(int).to_list(), id_map_df['movieId'].to_list()))
    return tmbd_to_movieid


class JSONData:
    def __init__(self, json_dict, id_map=None):
        self.tmbd_to_movieid = id_map
        self.json_dict = json_dict
        self.text_df = pd.DataFrame(json_dict).T.reset_index().drop('index', axis=1).sort_values(by=['id']).rename(
            columns={"id": "movieId"})
        self.text_df["movieId"] = self.text_df["movieId"].apply(lambda x: self.tmbd_to_movieid[x])
        self.text_df["release_date"] = pd.to_datetime(self.text_df["release_date"])
        self.text_df["release_date"] = (self.text_df["release_date"] - self.text_df["release_date"].min()) / np.timedelta64(1, 'Y')

    def process_context(self):
        # return a df with m_movies, and overview and tag_line columns
        json_context = self.text_df[['movieId', 'overview', 'tagline']]
        json_context = json_context.reset_index()
        return json_context

    def process_features(self):
        # return a df with m_movies, and other cate/num features in the json file
        # build a bow model to represent cast features
        cast = self.text_df[['movieId', 'cast']]
        cast = cast.assign(cast_split=cast["cast"].apply(lambda l: ", ".join(l.split("|"))))
        vectorizer = CountVectorizer()
        cast_vecs = vectorizer.fit_transform(cast['cast_split'])

        svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
        cast_df_pipe = make_pipeline(svd, Normalizer(copy=False))
        cast_feat = cast_df_pipe.fit_transform(cast_vecs)

        cast_feat_df = pd.DataFrame(cast_feat)
        json_feat = pd.concat(
            [self.text_df[['popularity', 'runtime', 'release_date', 'vote_average', 'vote_count', 'movieId']],
             cast_feat_df],
            axis=1)

        return json_feat


class CSVData:
    def __init__(self, csv_dict):
        self.csv_dict = csv_dict
        genre_list = ['Film-Noir',
                      'Animation',
                      'Crime',
                      'Horror',
                      '(no genres listed)',
                      'Sci-Fi',
                      'Romance',
                      'War',
                      'Documentary',
                      'Comedy',
                      'Drama',
                      'Western',
                      'Children',
                      'Action',
                      'Mystery',
                      'Musical',
                      'Fantasy',
                      'Thriller',
                      'IMAX',
                      'Adventure']
        self.genre_id_to_genre = {j: g for j, g in enumerate(genre_list)}
        self.genre_to_genre_id = {g: j for j, g in enumerate(genre_list)}
        self.genre_mat = None
        self.genre_dict = {}
        self.user_movie_map = None

    def process_movie_csv_data(self):
        # do two different joins, one for user, one for movie
        # for user table, we want to have n_user x k_user_feat
        # for movie table, we want to have m_movie x l_movie_feat
        ratings_df = self.csv_dict['ratings']
        movie_df = ratings_df.pivot(index='movieId', columns='userId', values='rating')
        movie_df.fillna(0, inplace=True)

        # then do svd on the movie_df to get movie feature matrix based on ratings
        svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
        movie_svd = make_pipeline(svd, Normalizer(copy=False))
        movie_feat = movie_svd.fit_transform(movie_df)
        movie_feat = pd.DataFrame(movie_feat)
        movie_feat['movieId'] = movie_df.index
        # print(f"movie_feat size: {movie_feat.shape}")

        # creating tag feature
        tag_df = self.csv_dict['tags']
        tag_df['merged_tag'] = tag_df.groupby(["movieId"], as_index=False)['tag'].transform(lambda x: '; '.join(x))
        tag_df = tag_df[['movieId', 'userId', 'merged_tag']].drop_duplicates(subset=['movieId', 'merged_tag'])

        # use svd again after getting tag tf-idf
        vectorizer = TfidfVectorizer(
            max_df=0.5,
            min_df=5,
            stop_words="english",
        )
        movie_tfidf = vectorizer.fit_transform(tag_df['merged_tag'])
        movie_text_feat = movie_svd.fit_transform(movie_tfidf)
        movie_text_feat = pd.DataFrame(movie_text_feat)
        movie_text_feat['movieId'] = tag_df['movieId']

        # print(f"movie_text_feat size: {movie_text_feat.shape}")

        # join with movie_feat df
        movie_feat = movie_feat.merge(movie_text_feat, how='left', left_on='movieId', right_on='movieId')
        # print(f"movie_feat size: {movie_feat.shape}")
        return movie_feat

    def process_movie_genre(self):
        # create movie features based on genre
        genre = self.csv_dict['movies']
        self.genre_mat = np.zeros([genre.shape[0], 20])

        for j, g in enumerate(genre['genres'].to_list()):
            for g_ in g.split('|'):
                self.genre_mat[j, self.genre_to_genre_id[g_]] = 1
            self.genre_dict[genre['movieId'][j]] = self.genre_mat[j,:]

        genre_df = pd.DataFrame([])
        genre_df['movieId'] = genre['movieId']
        for col in self.genre_to_genre_id:
            genre_df[col] = self.genre_mat[:, self.genre_to_genre_id[col]]

        genre_df = genre_df.reset_index()

        return genre_df

    def process_user_csv_data(self):
        ratings_df = self.csv_dict['ratings']
        # get the user df
        user_df = ratings_df.pivot(index='userId', columns='movieId', values='rating')

        user_feat = pd.DataFrame([])
        # get the rating distribution features
        for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
            column_title = f'pct_{str(int(q * 100))}'
            user_feat[column_title] = user_df.quantile(q, axis=1)
        # get the rating count feature
        user_feat['no_ratings'] = user_df.count(axis=1)

        user_feat = user_feat.reset_index()
        # user_df['userId'] = user_df.index
        user_df = user_df.reset_index()

        # get user watched movie history features. Num of movies watched per genre
        user_movie = pd.DataFrame([])
        user_movie['movie'] = ratings_df.groupby(["userId"], as_index=False)["movieId"]
        user_movie['userId'] = user_df["userId"]
        self.user_movie_map = dict(user_movie['movie'].to_list())
        user_movie_mat = np.zeros([len(self.user_movie_map), len(self.genre_to_genre_id)])
        for user in self.user_movie_map:
            user_mat = np.zeros([len(self.user_movie_map[user]), len(self.genre_to_genre_id)])
            for i, mid in enumerate(self.user_movie_map[user]):
                user_mat[i, :] = self.genre_dict[mid]
            user_movie_mat[user - 1] = np.sum(user_mat, axis=0) / len(self.user_movie_map[user])  # user is 1 idx
        user_movie_df = pd.DataFrame(user_movie_mat).add_prefix('user_')
        user_movie_df['userId'] = user_movie['userId']

        user_feat = user_feat.merge(user_movie_df, how='left', on='userId')

        return user_feat

    def create_id_map(self):
        id_map_df = self.csv_dict['links'].dropna()
        tmbd_to_movieid = dict(zip(id_map_df['tmdbId'].astype(int).to_list(), id_map_df['movieId'].to_list()))
        return tmbd_to_movieid
