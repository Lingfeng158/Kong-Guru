import os
import numpy as np

# postprocess_general_v2: same as _general, but with updated fanCalcLib.so for faster processing


def load_from_result(dst, file):
    dst_path = os.path.join(dst, file)
    with open(dst_path, "rb") as f:
        init_dist = np.load(f, allow_pickle=True)
        play_count = np.load(f, allow_pickle=True)
        play_to_shanten = np.load(f, allow_pickle=True)

        # meta
        winner_id = np.load(f, allow_pickle=True)[0]
        fan_sum = np.load(f, allow_pickle=True)[0]
        score = np.load(f, allow_pickle=True)
        fan_list = np.load(f, allow_pickle=True)
    return (
        init_dist,  # 1
        play_count,  # 2
        play_to_shanten,  # 3
        winner_id,  # 4
        fan_sum,
        score,
        fan_list,  # 5
    )
