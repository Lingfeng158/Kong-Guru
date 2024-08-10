import os
import numpy as np

# Postprocessing_alt is specifically calculating 起始最佳作牌方向 && 最终作牌方向
# Through finding least dist combinations at the beginning and at the end
# For 4 players


def load_from_result_alt(dst, file):
    dst_path = os.path.join(dst, file)
    with open(dst_path, "rb") as f:
        init_dist = np.load(f, allow_pickle=True)
        init_list_tier = np.load(f, allow_pickle=True)
        final_retro_dist = np.load(f, allow_pickle=True)
        # final_dist_comb = np.load(f, allow_pickle=True)
        final_list_tier = np.load(f, allow_pickle=True)
        consistent_tile_count = np.load(f, allow_pickle=True)
        others_tile_count = np.load(f, allow_pickle=True)
        intersection_related = np.load(f, allow_pickle=True)
        # meta
        winner_id = np.load(f, allow_pickle=True)[0]
    return (
        init_dist,  # 1
        init_list_tier,  # 2
        final_retro_dist,  # 3
        final_list_tier,  # 4
        consistent_tile_count,  # 5
        others_tile_count,  # 6
        intersection_related,  # 7
        winner_id,  # 8
    )
