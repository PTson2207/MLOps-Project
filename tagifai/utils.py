import json
import numpy as np
import random
from urllib.request import urlopen

def load_json_from_url(url):
    """
    Load Json data from a URL
    """
    data = json.loads(urlopen(url).read())
    return data

def load_dict(filepath):
    """Load a dictionary from a JSON filepath"""
    with open(filepath, "r") as f:
        data = json.load(f)
    return data 

def save_dict(data, filepath, cls=None, sortkeys=False):
    """Save a dictionary to a specific location"""
    with open(filepath, "w") as f:
        json.dump(data, indent=2, fp=f, cls=cls, sort_keys=sortkeys)

def set_seeds(seed=42):
    """Set seed for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
