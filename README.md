# Movie Genre API

This repo contains a Flask Rest API with two endpoints, both of them are POST method.

1. post_csv_and_train method.
2. post_csv_and_predict method.

### post_csv_and_train

This endpoint is used to post a csv with the following columns movie_id, year, synopsis and genres to train a model based on the year of the film and the synopsis applying NLP techniques.

### post_csv_and_predict

This endpoint is used to post a csv with the following columns movie_id, year and synopsis to predict the top 5 genres of a given film.
