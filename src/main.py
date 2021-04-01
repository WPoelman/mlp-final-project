#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
File name:   main.py
Authors:     Joëlle Bosman (s3794717)
             Larisa Bulbaai (s3651258)
             Wessel Poelman (s2976129)
Date:        01-04-2021
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

import spacy
from nltk import download, word_tokenize
from nltk.classify import accuracy
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import stopwords
from nltk.metrics import precision, recall
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# Download the nltk stopwords and wordnet corpus if needed
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

    nlp = spacy.load(SPACY_MODEL, disable=['tagger', 'ner'])

    dataset = []

    # The translation and set are both a lot faster O(1) when compared to
    # checking a list or string O(n).
    punct_translation = str.maketrans('', '', punctuation)
    stoplist = set(stopwords.words('english'))

    # ps = PorterStemmer()
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

            tokenized = []
            for token in word_tokenize(review_text):
                lemmatized = lem.lemmatize(token)
                if lemmatized not in stoplist:
                    tokenized.append(lemmatized)

            # tokenized = [
                # Wordstemming -> SVM accuracy 0.815284 &  KNN accuracy 0.783638
                # ps.stem(token) for token in word_tokenize(review_text)

                # LEMMATIZATION -> SVM accuracy 0.810830 &  KNN accuracy 0.786217
                #  for token in word_tokenize(review_text)

                # token for token in word_tokenize(review_text)

            #     if token not in stoplist
            # ]

            # bag_of_words = ({t: True for t in tokenized}, rating_label)

            dataset.append(
                {
                    # 'bag_of_words': bag_of_words,
                    # 'review_text': row[4],
                    'tokenized': tokenized,
                    # Change rating_label with rating to use non-grouped
                    # ratings, i.e. the 1-5 ratings
                    'rating_label': rating_label,
                    'year_month': row[2],
                    'reviewer_location': row[3],
                    'disneyland_location': row[5],
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


def evaluation_sklearn_model(model, test_examples, test_labels):
    y_pred = model.predict(test_examples)
    prec = precision_score(test_labels, y_pred, average='macro')
    rec = recall_score(test_labels, y_pred, average='macro')
    acc = accuracy_score(test_labels, y_pred)
    f = f1_score(test_labels, y_pred, average='macro')

    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}\nRecall: {rec}\nF-score: {f}")
    print("Confusion matrix:")
    print(confusion_matrix(test_labels, y_pred))


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

    only_vec_train = [i['doc_vector'] for i in train_feats]
    only_label_train = [j['rating_label'] for j in train_feats]

    only_vec_test = [i['doc_vector'] for i in test_feats]
    only_label_test = [j['rating_label'] for j in test_feats]

    print('\n-- BASELINE --\n')
    baseline = DummyClassifier()
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

    # First only use 'bag of words' as a feature
    # only_bow_test = [item[1] for item in test_feats]
    # only_bow_train = [item[1] for item in train_feats]

    # svm_classifier = train_svm(only_bow_train)
    # evaluation(svm_classifier, only_bow_test, LABELS)

    # knn_classifier = train_knn(only_bow_train)
    # evaluation(knn_classifier, only_bow_test, LABELS)


if __name__ == "__main__":
    main()
