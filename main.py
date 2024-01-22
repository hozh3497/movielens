from fastapi import FastAPI
from scripts.ranker import Ranker

import uvicorn


app = FastAPI(title="Movie Recommendation", description="API for recommending movies to maximize experience", version="1.0")

ranker = Ranker(json_path='data/tmdb', csv_path='data/csv')

@app.get("/")
async def root():
    return {"message": "Movie Recommendation api is running"}


@app.post('/predict', tags=["predictions"])
async def get_recommendation(event: dict):
    try:
        input_id = event.get('body')
        num_recommended = event.get('num_rec')
        recommendations = ranker.make_recommendation(input_id, n_sample=num_recommended)
        output_dict = {}
        for score, movie_id in recommendations:
            output_dict[movie_id] = {"name": ranker.user_features.id_to_movie[movie_id],
                                     "score": str(score)}

        return {
            'statusCode': 200,
            'body': {
                'parameters': output_dict
            }
        }
    except Exception as e:
        # Return the error message
        return {
            'statusCode': 400,
            'body': {
                'error': str(e)
            }
        }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
