"""
get_tweets_predict.py

usage: get_tweets_predict.py [-h] [--sarcastic_path SARCASTIC_PATH]
                     [--non_sarcastic_path NON_SARCASTIC_PATH]
                     [--log_path LOG_PATH]

Query twitter API for tweets over last 7 days, check for unique new tweets, and
predict on those unique new tweets using the best classifiers from training. Reports
results in logs and with graph.

optional arguments:
  -h, --help            show this help message and exit
  --sarcastic_path SARCASTIC_PATH
                        path to directory where results w/ #sarcasm should be
                        saved. Needs trailing "/"
  --serious_path SERIOUS__PATH
                        path to directory where results w/o #sarcasm should be
                        saved. Needs trailing "/"
  --log_path LOG_PATH   path to save log. Needs trailing "/"
"""

import os
import sys
import json
import argparse
import logging
import glob
import pickle

from datetime import datetime, timedelta
import numpy as np

from TwitterSearch import *
from login import *
from json_io import list_to_json, list_from_json
from ml import BEST_TRAINED_CLASSIFIERS, predictMultiple

FN_UNIQUE = "unique.json"
FN_HASH = "hash_dict.json"

if __name__ == "__main__":
    # Setup CLA parser
    parser = argparse.ArgumentParser(
        description='''Query twitter API for tweets over last 7 days, check for
        unique new tweets, and predict using best classifiers.'''
    )
    parser.add_argument(
        '--sarcastic_path', help='''path to directory where results
         w/ #sarcasm should be saved. Needs trailing "/"'''
    )
    parser.add_argument(
        '--serious_path', help='''path to directory
        where results w/o #sarcasm should be saved. Needs trailing "/"'''
    )
    parser.add_argument('--log_path', help='path to save log. Needs trailing "/"')
    # Parse CLAs
    args = parser.parse_args()
    # start and end date (for file naming/logging)
    end_date = datetime.strftime(datetime.now(), "%Y-%m-%d")
    start_date =  datetime.strftime( (datetime.now() - timedelta(days=7)), "%Y-%m-%d")
    filename = "{}_{}".format(start_date, end_date)
    # setup logger
    if args.log_path:
        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path)
        logger = logging.getLogger('root')
        FORMAT = "[%(asctime)s - %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
        logging.basicConfig(
            filename=args.log_path + filename + ".log",
            filemode='a',
            level=logging.INFO,
            format=FORMAT
        )
    # lists to store tweets
    sarcastic_tweets = []
    serious_tweets = []
    # create search orders
    if args.sarcastic_path:
        sarcastic_tso = TwitterSearchOrder()
        sarcastic_tso.set_keywords(['#sarcasm']) # query only tweets containing #sarcasm
        sarcastic_tso.set_language('en')
        sarcastic_tso.set_include_entities(True)
        sarcastic_tso.arguments.update({"tweet_mode": "extended"})
    if args.serious_path:
        serious_tso = TwitterSearchOrder()
        serious_tso.set_keywords(["-#sarcasm"]) # query tweets w/o #sarcasm
        serious_tso.set_language('en')
        serious_tso.set_include_entities(True)
        serious_tso.arguments.update({"tweet_mode": "extended"})
    # query twitter API and populate tweet lists
    try:
        ts = TwitterSearch(
            consumer_key = CONSUMER_KEY,
            consumer_secret = CONSUMER_SECRET,
            access_token = ACCESS_TOKEN,
            access_token_secret = ACCESS_SECRET
         )
        if args.sarcastic_path:
            for sarcastic_tweet in ts.search_tweets_iterable(sarcastic_tso):
                if not sarcastic_tweet['full_text'].lower().startswith('rt'):
                    sarcastic_tweets.append({
                        'id': sarcastic_tweet['id'],
                        'urls': not not sarcastic_tweet['entities']['urls'],
                        'media': "media" in sarcastic_tweet["entities"],
                        'text': sarcastic_tweet['full_text']})
        if args.serious_path:
            for serious_tweet in ts.search_tweets_iterable(serious_tso):
                if not serious_tweet['full_text'].lower().startswith('rt'):
                    serious_tweets.append({
                        'id': serious_tweet['id'],
                        'urls': not not serious_tweet['entities']['urls'],
                        'media': "media" in serious_tweet["entities"],
                        'text': serious_tweet['full_text']})
    except TwitterSearchException as e:
        logging.error(str(e))
    # write query results to log
    logging.info("Queried {} sarcastic tweets".format(len(sarcastic_tweets)))
    logging.info("Queried {} serious tweets".format(len(serious_tweets)))
    # Filter for only unique new tweets
    paths, tweets, X, y =  [], [], [], []
    if args.serious_path:
        paths.append(args.serious_path)
        tweets.append(serious_tweets)
    if args.sarcastic_path:
        paths.append(args.sarcastic_path)
        tweets.append(sarcastic_tweets)
    for path, ts in zip(paths, tweets):
        tweet_dataset = list_from_json(path + FN_UNIQUE, old_format=False)
        original_length_tweet_dataset = len(tweet_dataset)
        hash_dict = {}
        if os.path.exists(path + FN_HASH):
            hash_dict = list_from_json(path + FN_HASH)
        for tweet in ts:
            if str(tweet['id']) not in hash_dict:
                hash_dict[str(tweet['id'])] = True
                tweet_dataset.append(tweet)
                X.append(tweet['text'])
                y.append(True if path == args.sarcastic_path else False)
        # Save updated unique tweets list and hash dict
        list_to_json(tweet_dataset, path + FN_UNIQUE, old_format=False)
        logging.info(
            "Found and saved {} new unique {} tweets".format(len(tweet_dataset) - original_length_tweet_dataset,
             "sarcastic" if path == args.sarcastic_path else "serious")
        )
        list_to_json(hash_dict, path + FN_HASH)
    # load best classifiers and predict on new tweets
    best_classifiers = pickle.load(open(BEST_TRAINED_CLASSIFIERS, 'rb'))
    names = ["NB_unbal", "NB_bal", "LOG_unbal", "LOG_bal"]
    # test on balanced dataset if queried both sarcastic and serious tweets
    if args.serious_path and args.sarcastic_path:
        y = np.array(y)
        if len(y[y==True] < len(y[y==False])):
            # less unique sarcastic tweets than serious
            X = np.append(X[:len(y[y==True])], X[len(y[y==False]):])
            y = np.append(y[:len(y[y==True])], y[len(y[y==False]):])
        else:
            # more unique sarcastic tweets than serious
            X = np.append(X[:len(y[y==False])], X[len(y[y==True]):])
            y = np.append(y[:len(y[y==False])], y[len(y[y==True]):])
    logging.info("Testing on {} tweets".format(len(y)))
    results = predictMultiple(
        X,
        [best_classifiers[name]['classifier'] for name in names],
        names,
        [best_classifiers[name]['dvp'] for name in names],
        y,
        graph=True
    )
    # report results to log
    for name in names:
        logging.info(
            '''Results for {}:
            Score: {}
            Sarcastic, classified sarcastic: {}
            Sarcastic, classified serious: {}
            Serious, classified serious: {}
            Serious, classified sarcastic: {}
            '''.format(
            name,
            results[name]['score'],
            results[name]['trueSar'],
            results[name]['falseSer'],
            results[name]['trueSer'],
            results[name]['falseSar'])
        )
