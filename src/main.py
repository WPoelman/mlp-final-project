#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Filename:    main.py
Authors:     Joëlle Bosman  (s3794717)
             Larisa Bulbaai (s3651258)
             Wessel Poelman (s2976129)
Date:        08-04-2021
Repository:  https://github.com/WPoelman/mlp-final-project
Dataset:     https://www.kaggle.com/arushchillar/disneyland-reviews
Description: This script reads, cleans and splits a dataset of Disneyland
             reviews, with this data it trains en tests classification models.
             Note: this is the final 'best' system without all experimentation
             steps. This is why some things are not used or ignored. For more
             information about the experimentation, see the report or the
             (rather messy) notebook in the above mentioned repository.

Usage:       python <path_to_disneyland_dataset.csv>
"""
import csv
import os
import pickle
import random
import sys
from string import punctuation

import spacy
from nltk import download, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# Downloads the nltk stopwords and wordnet corpus for lemmatization (if needed)
download('stopwords')
download('wordnet')

# Program constants
CACHE_PATH = 'dataset_cache.pickle'
SPACY_MODEL = 'en_core_web_md'
LABELS = ['positive', 'indifferent', 'negative']


def create_dataset(filepath, use_cache=False):
    ''' Reads and cleans the csv file and structures the data '''

    print('Reading in files...')

    # Return early when a cache exists
    if use_cache and os.path.isfile(CACHE_PATH):
        with open(CACHE_PATH, 'rb') as cache:
            print('Used cached dataset!')
            return pickle.load(cache)

    print('Loading spacy model...')
    nlp = spacy.load(SPACY_MODEL, disable=['tagger', 'ner'])

    dataset = []

    # The translation and set are both a lot faster when compared to checking
    # a list or string.
    # punct_translation = str.maketrans('', '', punctuation)
    stoplist = set(stopwords.words('english'))

    print('Loading lemmatizer...')
    lem = WordNetLemmatizer()

    with open(filepath, 'r', encoding='latin-1') as f:
        reader = csv.reader(f, delimiter=",", )

        # Skip the header row
        next(reader, None)

        # Items per row:
        #   0 -> review id
        #   1 -> rating between 1-5
        #   2 -> year and month
        #   3 -> location of reviewer
        #   4 -> review text
        #   5 -> Disneyland location
        for index, row in enumerate(reader):
            if index != 0 and index % 1000 == 0:
                print(f'Working... {index} rows done so far')
            rating = int(row[1])

            if rating < 3:
                rating_label = 'negative'
            elif rating == 3:
                rating_label = 'indifferent'
            else:
                rating_label = 'positive'

            review_text = row[4].lower().strip()

            tokenized = []

            for token in word_tokenize(review_text):
                lemmatized = lem.lemmatize(token)
                if lemmatized not in stoplist:
                    tokenized.append(lemmatized)

            year, month = None, None

            if row[2] != 'missing':
                # 2019-4 for example
                [year, month] = row[2].split('-')

            dataset.append(
                {
                    'tokenized': tokenized,
                    'rating_label': rating_label,
                    'year': year,
                    'month': month,
                    'reviewer_location': row[3],
                    'disneyland_location': row[5].replace('Disneyland_', ''),
                    'doc_vector': nlp(' '.join(tokenized)).vector
                }
            )

    with open(CACHE_PATH, 'wb') as p:
        pickle.dump(dataset, p)

    return dataset


def split_train_test(feats, split=0.8):
    ''' Creates test, train and dev splits from the given dataset '''
    random.Random(1).shuffle(feats)

    cutoff = int(len(feats) * split)
    tenpercent = int((len(feats) - cutoff) / 2)
    split = cutoff + tenpercent

    train_feats = feats[:cutoff]
    test_feats = feats[cutoff:split]
    dev_feats = feats[split:]

    print(f"  Training set: {len(train_feats)}")
    print(f"  Test set: {len(test_feats)}")
    print(f"  Development set: {len(dev_feats)}")

    return train_feats, test_feats, dev_feats


def evaluation_sklearn_model(model, test_examples, test_labels):
    y_pred = model.predict(test_examples)
    prec = precision_score(test_labels, y_pred, average='macro')
    rec = recall_score(test_labels, y_pred, average='macro')
    acc = accuracy_score(test_labels, y_pred)
    f = f1_score(test_labels, y_pred, average='macro')

    print(f"Accuracy: {acc}\nPrecision: {prec}\nRecall: {rec}\nF-score: {f}")
    print(f"Confusion matrix:\n{confusion_matrix(test_labels, y_pred)}")


def main():
    try:
        file_path = sys.argv[1]
    except IndexError:
        print(__doc__)
        sys.exit(1)

    # Set the second argument to True to load a cached version of the dataset,
    # for testing this is a lot faster.
    dataset = create_dataset(file_path)
    train_feats, test_feats, dev_feats = split_train_test(dataset)

    only_vec_train = [i['doc_vector'] for i in train_feats]
    only_label_train = [j['rating_label'] for j in train_feats]

    only_vec_test = [i['doc_vector'] for i in test_feats]
    only_label_test = [j['rating_label'] for j in test_feats]

    print('\n-- BASELINE --\n')
    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(only_vec_train, only_label_train)
    evaluation_sklearn_model(baseline, only_vec_test, only_label_test)

    print('\n-- SVM --\n')
    svm = LinearSVC()
    svm.fit(only_vec_train, only_label_train)
    evaluation_sklearn_model(svm, only_vec_test, only_label_test)

    print('\n-- KNN --\n')
    knn = KNeighborsClassifier()
    knn.fit(only_vec_train, only_label_train)
    evaluation_sklearn_model(knn, only_vec_test, only_label_test)

    print('\n-- Decision Tree --\n')
    dtc = DecisionTreeClassifier()
    dtc.fit(only_vec_train, only_label_train)
    evaluation_sklearn_model(dtc, only_vec_test, only_label_test)


if __name__ == "__main__":
    main()
