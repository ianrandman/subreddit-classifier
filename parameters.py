"""
This file contains the parameters to search over for finding the best hyperparameters. The file also contains the
best hyperparameters that were found for each classifier (which is done in find_hyperparameters.py).

__author__ = Ian Randman
__author__ = David Dunlap
"""

import numpy as np

random_forest_max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
random_forest_max_depth.append(None)
params_to_search = {
                    'MultinomialNB':
                      {'clf__alpha': np.logspace(-5, 0, num=6),
                       'clf__fit_prior': [True, False],
                       'tfidf__norm': ['l1', 'l2', None],
                       'tfidf__use_idf': [True, False],
                       'tfidf__sublinear_tf': [True, False],
                       'vect__max_df': [0.50, 0.75, 1.0],
                       'vect__ngram_range': [(1, 1), (1, 2)]
                       },
                    'SVC':
                      {'clf__C': np.logspace(-2, 2, num=5),
                       'clf__gamma': np.logspace(-2, 2, num=5),
                       'clf__kernel': ['rbf', 'linear'],
                       'tfidf__norm': ['l1', 'l2', None],
                       'tfidf__use_idf': [True, False],
                       'tfidf__sublinear_tf': [True, False],
                       'vect__max_df': [0.50, 0.75, 1.0],
                       'vect__ngram_range': [(1, 1), (1, 2)]
                       },
                    'RandomForestClassifier':
                      {'clf__n_estimators': [int(x) for x in np.linspace(start=10, stop=2000, num=10)],
                       'clf__max_depth': random_forest_max_depth,
                       'clf__min_samples_split': [2, 5, 10],
                       'clf__min_samples_leaf': [1, 2, 4],
                       'clf__n_jobs': [-1],
                       'tfidf__norm': ['l1', 'l2', None],
                       'tfidf__use_idf': [True, False],
                       'tfidf__sublinear_tf': [True, False],
                       'vect__max_df': [0.50, 0.75, 1.0],
                       'vect__ngram_range': [(1, 1), (1, 2)]
                       },
                    'SGDClassifier':
                      {'clf__loss': ['hinge', 'log'],
                       'clf__penalty': ['l1', 'l2', None],
                       'clf__alpha': np.logspace(-4, 2, num=7),
                       'clf__n_jobs': [-1],
                       'tfidf__norm': ['l1', 'l2', None],
                       'tfidf__use_idf': [True, False],
                       'tfidf__sublinear_tf': [True, False],
                       'vect__max_df': [0.50, 0.75, 1.0],
                       'vect__ngram_range': [(1, 1), (1, 2)]
                       }
                    }

MultinomialNB_params = {'clf__alpha': 0.01,
                        'clf__fit_prior': True,
                        'tfidf__norm': 'l1',
                        'tfidf__use_idf': False,
                        'tfidf__sublinear_tf': True,
                        'vect__max_df': 1.0,
                        'vect__ngram_range': (1, 1)}

SVC_params = {'clf__C': 100.0,
              'clf__gamma': 0.1,
              'clf__kernel': 'rbf',
              'tfidf__norm': 'l2',
              'tfidf__use_idf': True,
              'tfidf__sublinear_tf': True,
              'vect__max_df': 0.5,
              'vect__ngram_range': (1, 2)}

RandomForestClassifier_params = {'clf__n_estimators': 894,
                                 'clf__max_depth': 100,
                                 'clf__min_samples_split': 2,
                                 'clf__min_samples_leaf': 1,
                                 'clf__n_jobs': -1,
                                 'tfidf__norm': 'l1',
                                 'tfidf__use_idf': True,
                                 'tfidf__sublinear_tf': True,
                                 'vect__max_df': 0.5,
                                 'vect__ngram_range': (1, 1)}

SGDClassifier_params = {'clf__loss': 'log',
                 'clf__penalty': 'l2',
                 'clf__alpha': 0.001,
                 'clf__n_jobs': -1,
                 'tfidf__norm': 'l2',
                 'tfidf__use_idf': True,
                 'tfidf__sublinear_tf': True,
                 'vect__max_df': 1.0,
                 'vect__ngram_range': (1, 2)}

classifier_name_to_params = {'MultinomialNB': MultinomialNB_params,
                             'SVC': SVC_params,
                             'RandomForestClassifier': RandomForestClassifier_params,
                             'SGDClassifier': SGDClassifier_params}

