import os
import numpy as np


# Analyzing randomness, with init-dist


def load_from_result(dst, file):
    dst_path = os.path.join(dst, file)
    with open(dst_path, "rb") as f:
        player_win_count_list = np.load(f, allow_pickle=True)
        player_score_list = np.load(f, allow_pickle=True)
        valid_list_length = np.load(f, allow_pickle=True)[0]
        total_list_length = np.load(f, allow_pickle=True)[0]
    return (
        player_win_count_list,
        player_score_list,
        valid_list_length,
        total_list_length,
    )
