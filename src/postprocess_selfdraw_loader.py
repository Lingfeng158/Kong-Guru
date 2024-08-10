import os
import numpy as np

# Postprocessing_alt is specifically calculating 起始最佳作牌方向 && 最终作牌方向
# Through finding least dist combinations at the beginning and at the end
# For 4 players

# Another Note: optimized version of postprocess_randomness_enhanced with C .so lib.


def load_from_result(dst, file):
    dst_path = os.path.join(dst, file)
    with open(dst_path, "rb") as f:
        init_dist = np.load(f, allow_pickle=True)
        init_hand_hash = np.load(f, allow_pickle=True)
        init_min_dist_hash = np.load(f, allow_pickle=True)
        final_min_dist_hash = np.load(f, allow_pickle=True)
        consistent_tile_count = np.load(f, allow_pickle=True)
        others_tile_count = np.load(f, allow_pickle=True)

        first_shanten = np.load(f, allow_pickle=True)
        first_hu = np.load(f, allow_pickle=True)
        # meta
        fan_sum = np.load(f, allow_pickle=True)[0]
        score = np.load(f, allow_pickle=True)
        winner_id = np.load(f, allow_pickle=True)[0]
    return (
        init_dist,  # 1
        init_hand_hash,  # 2
        init_min_dist_hash,  # 3
        final_min_dist_hash,  # 4
        consistent_tile_count,  # 5
        others_tile_count,  # 6
        first_shanten,  # 7
        first_hu,  # 8
        fan_sum,
        score,
        winner_id,  # 9
    )
