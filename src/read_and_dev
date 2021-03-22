import csv
import sys
from string import punctuation
from nltk.corpus import stopwords
from nltk import word_tokenize
import random


def create_dic(file):
    dic = {}
    stoplist = stopwords.words('english') + list(punctuation)
    with open(file) as dr:
        reader = csv.reader(dr, delimiter=",")
        for item in reader:
            info = ""
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
                    dic[item[0]].extend((text, rating, item[2], item[3], item[5]))
    return dic


def split_train_test(feats, split=0.8):
    dicToList = list(feats.items())
    random.Random(1).shuffle(dicToList)
    cutoff = int(len(feats) * split)
    tenpercent = int((len(feats) - cutoff) / 2)
    split = cutoff + tenpercent
    train_feats = dicToList[:cutoff]
    test_feats = dicToList[cutoff:split]
    dev_feats = dicToList[split:]

    print("  Training set: %i" % len(train_feats))
    print("  Test set: %i" % len(test_feats))
    print("  Development set: %i" % len(dev_feats))
    return train_feats, test_feats, dev_feats


def main():
    file = sys.argv[1]
    train_feats, test_feats, dev_feats = split_train_test(create_dic(file))

if __name__ == "__main__":
    main()
