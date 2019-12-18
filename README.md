# Subreddit Classifier

## Setup

To run, you must install the following dependencies:
```
pip install -U scikit-learn
pip install praw
pip install joblib
pip install nltk
```
Also, to ensure the natural language toolkit has access to the english stop words, make sure you run: 
```
nltk.download('stopwords')
```

In order to use predict_subreddit.py or train_and_evaluate.py, the "models" folder should be populated with each of the four models.

In order to use find_hyperparameters.py, a "temp" folder must be made.

## Usage

In order to simply test a classifier on a random reddit post, run:
```
python predict_subreddit.py
```

The --url flag can be used to specify a url of a post to predict.

Run the following to get more usage options:
```
python predict_subreddit.py -h
```

In order to train or evaluate a specific model, use train_and_evaluate.py.
Run the following to get more usage options:
```sh
python train_and_evaluate.py -h
```

For example, to train and save a model for Multinomial Naive Bayes, run:
```
python train_and_evaluate.py MultinomialNB -t
```

In order to train to find the best hyperparameters for given classifier, use find_hyperparameters.py.
Run the following to get more usage options:
```
python find_hyperparameters.py -h
```
Doing this will also train the best model on the full training set and save it.

Finally, if data still needs to be pulled from Reddit, run:
```
python get_data.py
```
This will attempt to pull 1000 posts from each subreddit, save them to a file, and save splits of the full data for 
training, development, and testing.
