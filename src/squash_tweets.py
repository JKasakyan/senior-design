"""
squash_tweets.py

USAGE:
$ python squash_tweets.py [-h] [--parse] path

Squash multiple jsons into one.

positional arguments:
  path        path to directory containing tweet jsons. Needs trailing "/"

optional arguments:
  -h, --help  show this help message and exit
  --parse     use if jsons to squash need to be parsed first
"""

import sys
import os
import glob
import argparse

from datetime import datetime

from json_io import list_to_json, list_from_json


if __name__ == "__main__":
    # Setup CLA parser
    parser = argparse.ArgumentParser(description='Squash multiple jsons into one')
    parser.add_argument('path', help='path to directory containing tweet jsons. Needs trailing "/"')
    parser.add_argument('--delete', action="store_true", help='delete old JSONs after squashing')
    # Parse CLAs
    args = parser.parse_args()
    if not os.path.exists(args.path):
        raise Exception("{} is not a valid path.".format(args.path))

    # Populate list with paths to jsons
    json_paths_lst = glob.glob(args.path + "*.json")

    # create one list of all merged tweets
    merged_lst = [ tweet for json_path in json_paths_lst for tweet in list_from_json(json_path)]

    # Get earliest and latest date of jsons squashed for naming purposes of merged file.
    parse_date_from_filename = lambda fn: fn.split('/')[-1].split('.')[0].split('_')
    sorted_dates = sorted([datetime.strptime(date, "%Y-%m-%d") for fn in json_paths_lst for date in parse_date_from_filename(fn)])
    from_date = datetime.strftime(sorted_dates[0], "%Y-%m-%d")
    to_date = datetime.strftime(sorted_dates[-1], "%Y-%m-%d")
    filename = "{}_{}.json".format(from_date, to_date)

    # Save merged list to json
    list_to_json(merged_lst, args.path + filename)

    # Delete old files if flag is set
    if args.delete:
        for json_path in json_paths_lst:
            os.remove(json_path)
