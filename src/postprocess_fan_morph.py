import os
import sys

sys.path.append(os.path.join(os.getcwd(), ".."))
sys.path.append(os.getcwd())
import postprocess_selfdraw_loader
import src_py.rule
import numpy as np
from multiprocessing import Pool
import json

# Postprocess_selfdraw_next is based on results from postprocess_selfdraw,
# further analyze fan-type morphing and connection between targets
# 番种之间的变化与关联
# Pre-requisite: processed_enhanced from postprocess_selfdraw


def load_from_result(dst, file):
    dst_path = os.path.join(dst, file)
    with open(dst_path, "rb") as f:
        init_dist = np.load(f, allow_pickle=True)
        init_final_target_diff = np.load(f, allow_pickle=True)
        tile_in_init_target = np.load(f, allow_pickle=True)
        tile_in_final_target = np.load(f, allow_pickle=True)
        tile_diff_between_targets = np.load(f, allow_pickle=True)
        init_fan_type_and_count = np.load(f, allow_pickle=True)

        final_fan_type_and_count = np.load(f, allow_pickle=True)
        # meta
        winner_id = np.load(f, allow_pickle=True)[0]
    return (
        init_dist,  # 1
        init_final_target_diff,  # 2
        tile_in_init_target,  # 3
        tile_in_final_target,  # 4
        tile_diff_between_targets,  # 5
        init_fan_type_and_count,  # 6
        final_fan_type_and_count,  # 7
        winner_id,  # 8
    )


def compute_min_diff_target(init_list, final_target):
    # find the min dst init-final target
    min_diff = 15
    min_diff_hash = None
    for init_target in init_list:
        abs_diff = [min(int(a), int(b)) for a, b in zip(init_target, final_target)]
        diff = 14 - sum(abs_diff)
        if diff < min_diff:
            min_diff = diff
            min_diff_hash = init_target

    return min_diff, min_diff_hash


def compute_hash_overlap(hash1, hash2):
    # return the overlap betwen two hashes
    hash_as_int = [min(int(a), int(b)) for (a, b) in zip(hash1, hash2)]
    # hash_as_int = []
    # for a, b in zip(hash1, hash2):
    #     hash_as_int.append(min(int(a), int(b)))
    return "".join(map(str, hash_as_int)), sum(hash_as_int)


def compute_hash_diff(hash1, hash2):
    # return the difference betwen two hashes
    # treating hash1 as the hash with more overlap, tile in hash2 but not in hash1 will not be counted
    hash_as_int = [max(int(a) - int(b), 0) for (a, b) in zip(hash1, hash2)]
    # hash_as_int = []
    # for a, b in zip(hash1, hash2):
    #     hash_as_int.append(min(int(a), int(b)))
    return "".join(map(str, hash_as_int)), sum(hash_as_int)


def postprocessing_fan_morph(path, dst, file):
    # fmt: off
    tile_list_raw = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9',  #饼
                'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9',   #万
                'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9',   #条
                'F1', 'F2', 'F3', 'F4', 'J1', 'J2', 'J3' #风、箭


    ]
    # fmt: on
    tile_list = {}
    for tile in tile_list_raw:
        tile_list[tile] = 4
    dst_path = os.path.join(dst, file)
    (
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
    ) = postprocess_selfdraw_loader.load_from_result(path, file)
    # print(init_min_dist_hash)

    init_final_target_diff = []
    tile_in_init_target = []
    tile_in_final_target = []
    tile_diff_between_targets = []
    init_fan_type_and_count = []
    final_fan_type_and_count = []

    for id in range(4):
        id_init_hand_hash = init_hand_hash[id][0]
        id_init_min_dist_hash = init_min_dist_hash[id]
        id_final_min_dist_hash = final_min_dist_hash[id][0]
        if id_final_min_dist_hash == "":
            init_final_target_diff.append(99)
            tile_in_init_target.append(0)
            tile_in_final_target.append(0)
            tile_diff_between_targets.append("")
            init_fan_type_and_count.append([])
            final_fan_type_and_count.append([])
        else:
            min_diff, min_diff_init_hash = compute_min_diff_target(
                id_init_min_dist_hash, id_final_min_dist_hash
            )
            init_target_overlap, init_overlap_count = compute_hash_overlap(
                id_init_hand_hash, min_diff_init_hash
            )
            final_target_overlap, final_overlap_count = compute_hash_overlap(
                id_init_hand_hash, id_final_min_dist_hash
            )
            diff_between_overlaps, count_diff = compute_hash_diff(
                init_target_overlap, final_target_overlap
            )

            init_final_target_diff.append(min_diff)
            tile_in_init_target.append(init_overlap_count)
            tile_in_final_target.append(final_overlap_count)
            tile_diff_between_targets.append(diff_between_overlaps)
            # calc fan with rule.calc_exact_fan_with_PyMahJongGB
            final_hand = src_py.rule.restore_hashed_tiles(id_final_min_dist_hash)
            final_win_tile = list(final_hand.keys())[0]
            # print(final_hand)
            if sum(final_hand.values()) == 16:
                final_fan_type = ["全不靠"]
            else:
                (
                    final_fan_sum,
                    final_fan_type,
                ) = src_py.rule.calc_exact_fan_with_PyMahJongGB(
                    [], final_hand, final_win_tile, False, False, False, False, id, 0
                )
            if "单钓将" in final_fan_type:
                final_fan_type.remove("单钓将")
            if "嵌张" in final_fan_type:
                final_fan_type.remove("嵌张")
            if "边张" in final_fan_type:
                final_fan_type.remove("边张")
            if "门前清" in final_fan_type:
                final_fan_type.remove("门前清")

            init_hand = src_py.rule.restore_hashed_tiles(min_diff_init_hash)
            init_win_tile = list(init_hand.keys())[0]
            # print(init_hand)
            if sum(init_hand.values()) == 16:
                init_fan_type = ["全不靠"]
            else:
                (
                    init_fan_sum,
                    init_fan_type,
                ) = src_py.rule.calc_exact_fan_with_PyMahJongGB(
                    [], init_hand, init_win_tile, False, False, False, False, id, 0
                )
            if "单钓将" in init_fan_type:
                init_fan_type.remove("单钓将")
            if "嵌张" in init_fan_type:
                init_fan_type.remove("嵌张")
            if "边张" in init_fan_type:
                init_fan_type.remove("边张")
            if "门前清" in init_fan_type:
                init_fan_type.remove("门前清")

            init_fan_type_and_count.append(init_fan_type)
            final_fan_type_and_count.append(final_fan_type)

    with open(dst_path, "wb") as f:
        # init distance
        np.save(f, np.array(init_dist))
        # difference between initial target and final target (value)
        np.save(f, np.array(init_final_target_diff))
        # initial hand tiles in initial target
        np.save(f, np.array(tile_in_init_target))
        # initial hand tiles in final target
        np.save(f, np.array(tile_in_final_target))
        # initial hand tiles in initial target, but not in final target
        np.save(f, np.array(tile_diff_between_targets))
        # initial fan type
        np.save(f, np.array(init_fan_type_and_count, dtype=object))
        # final fan type
        np.save(f, np.array(final_fan_type_and_count, dtype=object))
        # winner id
        np.save(f, np.array([winner_id]))


if __name__ == "__main__":
    cpuCount = os.cpu_count() - 2
    path = "processed_enhanced"
    dst = "processed_fan_morph"
    if not os.path.isdir(dst):
        os.makedirs(dst)
    # file = "61611.npy"
    # postprocessing_fan_morph(path, dst, file)
    dir_list = os.listdir(path)

    # file_list = os.listdir(dst)
    # unprocessed_list = []
    # for file in dir_list:
    #     if file not in file_list:
    #         unprocessed_list.append(file)
    # dir_list = unprocessed_list

    dir_list.sort()
    pool = Pool(cpuCount)
    for fil in dir_list:
        # for fil in file_unprocessed:
        pool.apply_async(
            postprocessing_fan_morph,
            args=(path, dst, fil),
        )
    pool.close()
    pool.join()
    print("Post Processing Done!")
