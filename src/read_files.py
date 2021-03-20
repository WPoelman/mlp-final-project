import csv
import sys
from string import punctuation
from nltk.corpus import stopwords
from nltk import word_tokenize
import random
import itertools


def create_dic(file):
    file = sys.argv[1]
    dic = {}
    stoplist = stopwords.words('english') + list(punctuation)
    with open(file) as dr:
        reader = csv.reader(dr, delimiter=",")
        for item in reader:
            rate = (item[1])
            if rate in ['1', '2', '3', '4', '5']:
                rate = int(rate)
                if rate < 3:
                    rating = 'negative'
                elif rate == 3:
                    rating = 'indifferent'
                else:
                    rating = 'positive'
                string = item[4].lower()
                tokenized = [token for token in word_tokenize(
                    string) if token not in stoplist]
                text = ' '.join(tokenized)
                if item[0] not in dic:
                    dic[item[0]] = []
                    dic[item[0]].append(text)
                    dic[item[0]].append(rating)

    return dic


def split_train_test(feats, split=0.8):

    dicToList = list(feats.items())
    cutoff = int(len(feats) * split)
    train_feats = dicToList[:cutoff]
    test_feats = dicToList[cutoff:]

    print("  Training set: %i" % len(train_feats))
    print("  Test set: %i" % len(test_feats))
    return train_feats, test_feats


def main():
    file = sys.argv[1]

    train_feats, test_feats = split_train_test(create_dic(file))


if __name__ == "__main__":
    main()
