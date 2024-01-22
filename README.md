**Movie Recommendation Engine**
This is a simple implementation of movie recommendation based on the MovieLens dataset. The goal is to making recommendations that maximizes the probability for a user to rent, rather than purchase, the recommended movie.

The recommendation aims to recommend movies to users already in the dataset.

The recommendation is made in 3 steps: first, for a given user id, a sample of movies that have been watched by the user were randomly selected. For each movie in the sample, k similar movies are selected based on textual features (as the overview and tagline fields in tmdb data) using KNN. The features related to users and movies, such as the distribution of user rating per user, the distribution of genres watched by the user, as well as movie features such as movie genre, movie cast, movie tags provided by users, were combined to feed into the rating prediction model. The final recommendation is made from among the selected movies from step one, ordered by the predicted wieghting. However, due to time constraint, not enough effort was able to put in making business rules which will serve as added scoring mechnism to select final candidates. Future work will be needed for this.

As for the embedding models, proposals for 1. An autoencoder architecture for learning an embedding model for movie text feature representation, and 2. An MLP based classfication model for rating prediction were made and code base implemented. However, since there wasn't enough time to properly train the model, the current deployment does not included those models. Rather, simpler tf-idf features and xgboost model were used in place of the additional architectural considerations.

***Usage***
The files should follow the following structure:
/
|- data/
     |- csv/
         |- *.csv
     |- tmdb/
         |- *.json
 |- scripts/
 |
 main.py

 note that the .zip file in ./data/tmdb should be uncompressed in the same directory for the program to work properly. You can install fastapi and uvicorn, in commandline in root directory, run: uvicorn main:app --reload and then go to http://127.0.0.1:8000/docs to play around with the api. The api input should follow this format:
 {
 "body": "user_id",
 "num_rec": "num_recommended_movies"
 }

 A docker file is also included in the repo. To run the endpoint with docker, run: docker build -t movie-recommender .   then docker run -d -p 8000:8000 --name fastapi-app movie-recommender and the endpoint should be accessible from http://127.0.0.1:8000/predict.
