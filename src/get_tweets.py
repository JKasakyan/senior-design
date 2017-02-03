import os
import sys
import json
from datetime import datetime, timedelta
import logging

from TwitterSearch import *
from login import *

'''
get_tweets.py

Retrieve a week's worth of tweets (one batch of #sarcastic, one w/ no specification)
and save to json.

USAGE:
$ python get_tweets.py
'''

def create_sarcastic_search_order():
    tso = TwitterSearchOrder()
    tso.set_keywords(['#sarcasm']) # query only tweets containing #sarcasm
    tso.set_language('en')
    tso.set_include_entities(True)
    return tso

def create_non_sarcastic_search_order():
    tso = TwitterSearchOrder()
    tso.set_keywords(['the', '-#sarcasm']) # must have keyword, so query tweets containing 'the' but NOT '#sarcasm'
    tso.set_language('en')
    tso.set_include_entities(True)
    return tso

if __name__ == "__main__":
    # default paths
    SARCASTIC_DIR = "../json/sarcastic/"    # path to store sarcastic tweet json
    NON_SARCASTIC_DIR = "../json/non_sarcastic/" # path to store non sarcastic tweet json
    LOGGING_DIR = "../json/logs/" # path to save log

    # start and end date (for file naming/logging)
    end_date = datetime.now()
    start_date =  end_date - timedelta(days=7)
    filename = "{}-{}-{}_{}-{}-{}".format(start_date.year, start_date.month, start_date.day, end_date.year, end_date.month, end_date.day)

    # setup logger
    if not os.path.exists(LOGGING_DIR):
        os.makedirs(LOGGING_DIR)
    logger = logging.getLogger('root')
    FORMAT = "[%(asctime)s - %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(filename=LOGGING_DIR + filename + ".log", filemode='a', level=logging.INFO, format=FORMAT)

    # lists to store tweets
    sarcastic_tweets_list = []
    non_sarcastic_tweets_list = []

    # create search orders
    sarcastic_tso = create_sarcastic_search_order()
    non_sarcastic_tso = create_non_sarcastic_search_order()

    try:
        # query twitter API and populate tweet lists
        ts = TwitterSearch(
            consumer_key = CONSUMER_KEY,
            consumer_secret = CONSUMER_SECRET,
            access_token = ACCESS_TOKEN,
            access_token_secret = ACCESS_SECRET
         )
        for sarcastic_tweet in ts.search_tweets_iterable(sarcastic_tso):
            sarcastic_tweets_list.append(sarcastic_tweet)
        for non_sarcastic_tweet in ts.search_tweets_iterable(non_sarcastic_tso):
            non_sarcastic_tweets_list.append(non_sarcastic_tweet)
    except TwitterSearchException as e:
        logging.error(str(e))

    # save results to json
    if not os.path.exists(SARCASTIC_DIR):
        os.makedirs(SARCASTIC_DIR)
    if not os.path.exists(NON_SARCASTIC_DIR):
        os.makedirs(NON_SARCASTIC_DIR)
    with open(SARCASTIC_DIR + filename + ".json", 'w') as f:
        json.dump(sarcastic_tweets_list, f)
        logging.info("Saved {} sarcastic tweets at {}".format(len(sarcastic_tweets_list), f.name))
    with open(NON_SARCASTIC_DIR + filename + ".json", 'w') as f:
        json.dump(non_sarcastic_tweets_list, f)
        logging.info("Saved {} non sarcastic tweets at {}".format(len(non_sarcastic_tweets_list), f.name))
