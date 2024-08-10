import json
import urllib.request
import os
from multiprocessing import Pool
import rule
import numpy as np


def load_from_result(dst, file):
    dst_path = os.path.join(dst, file)
    with open(dst_path, "rb") as f:
        init_dist = np.load(f, allow_pickle=True)
        first_hu = np.load(f, allow_pickle=True)
        first_shanten = np.load(f, allow_pickle=True)
        scores = np.load(f, allow_pickle=True)
    return (
        init_dist,  # 1
        first_hu,  # 2
        first_shanten,  # 3
        scores,  # 4
    )
