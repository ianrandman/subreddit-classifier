# Subreddit Classifier

##Setup

To run, you must install the following dependencies:
```sh
> pip install -U scikit-learn
> pip install praw
> pip install joblib
> pip install nltk
```
Also, to ensure the natural language toolkit has access to the english stop words, make sure you run: 
```sh
> nltk.download('stopwords')
```

In order to use predict_subreddit.py or train_and_evaluate.py, the "models" folder should be populated with each of the four models.

In order to use find_hyperparameters.py, a "temp" folder must be made.

##Usage

In order to simply test a classifier on a random reddit post, run:
```sh
> python predict_subreddit.py
```

Run the following to get more usage options:
```sh
> python predict_subreddit.py -h
```

In order to train or evaluate a specific model, use train_and_evaluate.py.
Run the following to get more usage options:
```sh
> python train_and_evaluate.py -h
```

For example, to train and save a model for Multinomial Naive Bayes, run:
```sh
> python train_and_evaluate.py MultinomialNB -t
```

In order to train to find the best hyperparameters for given classifier, use find_hyperparameters.py.
Run the following to get more usage options:
```sh
> python find_hyperparameters.py -h
```
Doing this will also train the best model on the full training set and save it.

Finally, if data still needs to be pulled from Reddit, run:
```sh
> python get_data.py
```
This will attempt to pull 1000 posts from each subreddit, save them to a file, and save splits of the full data for 
training, development, and testing.