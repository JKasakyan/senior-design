"""
process_tweets.py

Template for processing tweets
"""


import glob
import argparse

from json_io import list_to_json, list_from_json


if __name__ == "__main__":
    # Setup CLA parser
    parser = argparse.ArgumentParser(description='Process tweets')
    parser.add_argument('path', help='path to directory containing tweet jsons. Needs trailing "/"')
    # parse CLAs
    args = parser.parse_args()
    JSON_DIR = args.path

    # Populate list with paths to jsons
    json_paths_lst = glob.glob(JSON_DIR + "*-*-*_*-*-*.json")

    # Process tweets
    # processed_tweets = [ process(tweet) for json_path in json_paths_lst for tweet in list_from_json(json_path)]

    # Save processed tweets to json
    # list_to_json(processed_tweets, path/to/save/tweets.json)
