"""
json_io.py

Functions related to reading/writing lists to json and json to lists
"""


import json


def list_from_json(json_file):
    """Return a list corresponding to contents of json file"""

    with open(json_file, 'r') as fp:
        return json.load(fp)

def list_to_json(lst, path):
    """Save a list to a json file at corresponding path"""

    with open(path, 'w') as fp:
        json.dump(lst, fp, sort_keys=True, indent=4)
