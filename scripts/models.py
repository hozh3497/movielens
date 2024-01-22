import pickle
import torch
from torch import nn

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb


class MLP(nn.Module):
    # the fully connected network component
    def __init__(self, input_dim,
                 hidden_dims,
                 output_dim,
                 dropout=0,
                 hidden_activation=nn.LeakyReLU,
                 output_activation=nn.Sigmoid):
        super(MLP, self).__init__()
        layer_dims = [input_dim] + list(hidden_dims) + [output_dim]
        assert all([l > 0 for l in layer_dims])
        mlp_layers = []
        # hidden layers
        for i in range(len(layer_dims) - 2):
            mlp_layers.append(
                LinearLayer(
                    in_features=layer_dims[i],
                    out_features=layer_dims[i + 1],
                    activation=hidden_activation,
                    dropout=dropout,
                )
            )
        # output layer
        mlp_layers.append(
            LinearLayer(
                in_features=layer_dims[-2],
                out_features=layer_dims[-1],
                activation=output_activation,
                dropout=0.0,
            )
        )
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        return self.mlp(x)


class LinearLayer(nn.Module):
    def __init__(self, in_features: int,
                 out_features: int,
                 activation: nn,
                 dropout: float):
        super().__init__()
        modules = [
            nn.Linear(in_features=in_features, out_features=out_features),
            activation(),
        ]
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout))
        self.layer = nn.Sequential(*modules)

    def forward(self, x, **kwargs):
        return self.layer(x)


class Encoder(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=None, embed_dim=None, dropout=0.2, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = dropout

        self.encoder = nn.Sequential(
            self.make_encoder_block(input_dim, hidden_dim),
            self.make_encoder_block(hidden_dim, embed_dim),
        )

    def make_encoder_block(self, input_dim, output_dim):
        return nn.Sequential(nn.Linear(input_dim, output_dim),
                             nn.BatchNorm1d(output_dim),
                             nn.ReLU(inplace=True),
                             nn.Dropout(self.dropout))

    def forward(self, x):
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self, embed_dim=None, hidden_dim=None, output_dim=None, dropout=0.2, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = dropout

        self.decoder = nn.Sequential(
            self.make_decoder_block(embed_dim, hidden_dim),
            self.make_decoder_block(hidden_dim, output_dim, final_layer=True),
        )

    def make_decoder_block(self, input_dim, output_dim, final_layer=False):
        if final_layer is False:
            return nn.Sequential(nn.Linear(input_dim, output_dim),
                                 nn.BatchNorm1d(output_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(self.dropout))
        return nn.Sequential(nn.Linear(input_dim, output_dim),
                             nn.BatchNorm1d(output_dim),
                             nn.Sigmoid(),
                             )

    def forward(self, x):
        x = self.decoder(x)
        return x


class ContentAutoEncoder(nn.Module):
    # use overview text to predict movie genre
    def __init__(self, input_dim, hidden_dim, embed_dim, output_dim, dropout=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim

        self.encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, embed_dim=embed_dim)
        self.decoder = Decoder(embed_dim=embed_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    @staticmethod
    def get_representation(dataset):
        # get the initial embedding representations with data in dataset
        return dataset.generate_features()

    def get_embedding(self, dataset):
        # only do the encoding pass to get the encoder representation
        embed_input = self.get_representation(dataset)
        embed = self.encoder(embed_input)
        return embed

    def forward(self, x):
        # encode the input text with lm and bow, and then get output using encoder/decoder
        embed_input = self.get_representation(dataset=x)
        embed = self.encoder(embed_input)
        y = self.decoder(embed)
        return y


class UserMLPModel(nn.Module):
    # Use an mlp to predict user ratings based on movie features and user features
    def __init__(self, input_dim,
                 hidden_dims,
                 output_dim,
                 dropout=0,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp = MLP(input_dim,
                       hidden_dims,
                       output_dim,
                       dropout=dropout, )
        self.linear = LinearLayer(output_dim,
                                  1,
                                  None,
                                  dropout)

    @staticmethod
    def get_representation(dataset):
        return dataset.generate_features()

    def get_embeddings(self, dataset):
        with torch.no_grad:
            model_input = self.get_representation(dataset=dataset)
            last_layer = self.mlp(model_input)
        return last_layer

    def forward(self, x):
        model_input = self.get_representation(dataset=x)
        last_layer = self.mlp(model_input)
        y = self.linear(last_layer)
        return y


class ContentKNNModel:
    def __init__(self,
                 metric='cosine',
                 algorithm='auto',
                 n_neighbors=20,
                 n_jobs=-1,
                 **kwargs,
                 ):
        self.knn = NearestNeighbors(metric=metric,
                                    algorithm=algorithm,
                                    n_neighbors=n_neighbors,
                                    n_jobs=n_jobs)

    def fit(self, dataset):
        self.knn.fit(dataset)

    def save_model(self, model_path):
        pickle.dump(self.knn, open(model_path + '/content_knn_model.pkl', 'wb'))

    def load_model(self, model_path):
        self.knn = pickle.load(open(model_path, 'rb'))

    def inference(self, movie_mat, top_k=20):
        distances, indices = self.knn.kneighbors(movie_mat, n_neighbors=top_k + 1)
        return list(zip(indices.squeeze().tolist(), distances.squeeze().tolist()))


class UserXGBModel:
    def __init__(self,
                 k_fold=3,
                 ):
        self.xgb = xgb.XGBRegressor(n_jobs=1)
        self.k_fold = k_fold
        self.error = []

    def fit(self, dataset, y_lab="rating", ignore_labs=None):
        if not ignore_labs:
            ignore_labs = ['userId', 'movieId', 'index']

        X = dataset.loc[:, dataset.columns[~dataset.columns.isin(ignore_labs + [y_lab])]]
        y = dataset[y_lab]
        X = X.to_numpy()
        y = y.to_numpy()

        if self.k_fold > 1:
            kf = KFold(n_splits=self.k_fold, shuffle=True, random_state=31)
            for train_index, test_index in kf.split(X):
                self.xgb.fit(X[train_index], y[train_index])
                predictions = self.xgb.predict(X[test_index])
                actuals = y[test_index]
                self.error.append(mean_squared_error(actuals, predictions))

        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            self.xgb.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="rmse", eval_set=[(X_test, y_test)])
            self.error = self.xgb.evals_result()

    def inference(self, x):
        x = x.to_numpy()
        return self.xgb.predict(x)

    def save_model(self, model_path):
        pass

    def load_model(self, model_path):
        pass
