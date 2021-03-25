#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
File name:   main.py

Authors:     Joëlle Bosman (s3794717)
             Larisa Bulbaai (s3651258)
             Wessel Poelman (s2976129)

Date:        25-03-2021

Description: This script reads, cleans and splits the Disneyland reviews
             dataset used for training the classification models.

Usage: python <path_to_disneyland_dataset.csv>
"""
import csv
import os
import pickle
import random
import sys
from collections import defaultdict
from string import punctuation

from nltk import download, word_tokenize
from nltk.classify import accuracy
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import stopwords
from nltk.metrics import precision, recall
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

# Download the nltk stopwords corpus if needed
download('stopwords')

# Program constants
CACHE_PATH = 'dataset_cache.pickle'
LABELS = ['positive', 'indifferent', 'negative']


def create_dataset(filepath, use_cache=False):
    ''' Reads and cleans the csv file and structures the data '''

    print('Reading in files...')

    # Return early when a cache exists
    if use_cache and os.path.isfile(CACHE_PATH):
        with open(CACHE_PATH, 'rb') as cache:
            print('Used cached dataset!')
            return pickle.load(cache)

    dataset = []

    # The translation and set are both a lot faster O(1) when compared to
    # checking a list or string O(n).
    punct_translation = str.maketrans('', '', punctuation)
    stoplist = set(stopwords.words('english'))

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
        for row in reader:
            rating = int(row[1])

            if rating < 3:
                rating_label = 'negative'
            elif rating == 3:
                rating_label = 'indifferent'
            else:
                rating_label = 'positive'

            review_text = row[4] \
                .translate(punct_translation) \
                .lower() \
                .strip()

            tokenized = [
                token for token in word_tokenize(review_text)
                if token not in stoplist
            ]

            bag_of_words = ({t: True for t in tokenized}, rating_label)

            dataset.append(
                (tokenized, bag_of_words, rating_label, row[2], row[3], row[5])
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

    print("  Training set: %i" % len(train_feats))
    print("  Test set: %i" % len(test_feats))
    print("  Development set: %i" % len(dev_feats))

    return train_feats, test_feats, dev_feats


def precision_recall(classifier, testfeats):
    ''' Taken from classification.py from assignment 1.
        Calculates precision and recall for a given classifier.
    '''
    refsets = defaultdict(set)
    testsets = defaultdict(set)

    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    precisions = {}
    recalls = {}

    for label in classifier.labels():
        precisions[label] = precision(refsets[label], testsets[label])
        recalls[label] = recall(refsets[label], testsets[label])

    return precisions, recalls


def calculate_f(precisions, recalls):
    ''' Taken from assignment 1 from Wessel.
        Calculates the F-score for given a category given the precision and
        recall scores for that category.
    '''
    f_measures = {}

    for category in precisions.keys():
        # This is done to prevent the program from crashing when
        # no measure is provided for a particular category
        if not precisions[category] or not recalls[category]:
            f_measures[category] = None
            continue

        f_measures[category] = round(
            2 * ((precisions[category] * recalls[category]) /
                 (precisions[category] + recalls[category])), 6)

    return f_measures


def evaluation(classifier, test_feats, categories):
    ''' Taken from assignment 1, calculates and prints evaluation measures '''

    print("\nEvaluation...")
    print("  Accuracy: %f" % accuracy(classifier, test_feats))
    precisions, recalls = precision_recall(classifier, test_feats)
    f_measures = calculate_f(precisions, recalls)

    print(" |-----------|-----------|-----------|-----------|")
    print(" |%-11s|%-11s|%-11s|%-11s|" %
          ("category", "precision", "recall", "F-measure"))
    print(" |-----------|-----------|-----------|-----------|")
    for category in categories:
        if precisions[category] is None:
            print(" |%-11s|%-11s|%-11s|%-11s|" % (category, "NA", "NA", "NA"))
        else:
            print(" |%-11s|%-11f|%-11f|%-11s|" %
                  (category,
                   precisions[category],
                   recalls[category],
                   f_measures[category])
                  )
    print(" |-----------|-----------|-----------|-----------|")


def train_svm(train_feats):
    ''' Trains and returns a linear SVM classifier '''
    print('\nTraining SVM classifier...')

    # dual=False makes this considerably faster and is appropriate since the
    # dataset has more samples than features. Either way it should get roughly
    # the same result.
    return SklearnClassifier(LinearSVC(dual=False)).train(train_feats)


def train_knn(train_feats):
    ''' Trains and returns a KNN classifier '''
    print('\nTraining KNN classifier...')

    return SklearnClassifier(KNeighborsClassifier()).train(train_feats)


def main():
    try:
        file_path = sys.argv[1]
    except IndexError:
        print(__doc__)
        sys.exit(1)

    # Set the second argument to True to load a cached version of the dataset,
    # for testing this is a lot faster.
    dataset = create_dataset(file_path, True)
    train_feats, test_feats, dev_feats = split_train_test(dataset)

    # First only use 'bag of words' as a feature
    only_bow_test = [item[1] for item in test_feats]
    only_bow_train = [item[1] for item in train_feats]

    svm_classifier = train_svm(only_bow_train)
    evaluation(svm_classifier, only_bow_test, LABELS)

    knn_classifier = train_knn(only_bow_train)
    evaluation(knn_classifier, only_bow_test, LABELS)

    ''' TODO:
        -   Word embeddings toevoegen en aan training data toevoegen en / of
            los trainen en vergelijken
        -   Andere features uit de dataset toevoegen en testen wat betere
            scores oplevert
        -   Experimenteren met parameters van modellen

    '''


if __name__ == "__main__":
    main()
