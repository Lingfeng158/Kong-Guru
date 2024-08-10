# Analyais on Fan 对番种之研究
import numpy as np
import os
from collections import defaultdict
from multiprocessing import Pool
import rule


def default_zero():
    return 0


# fmt: off
tile_list_raw = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9',  #饼
            'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9',   #万
            'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9',   #条
            'F1', 'F2', 'F3', 'F4', 'J1', 'J2', 'J3' #风、箭
]
# fmt: on


def test_validity(list_of_combination_dicts):
    """
    Test if any tile_comb exceed limit of 4
    """
    tile_holder = defaultdict(default_zero)
    for combination_dict in list_of_combination_dicts:
        for k in combination_dict:
            tile_holder[k] += combination_dict[k]
            if tile_holder[k] > 4:
                return False
    return True


# Brute-force through all possible combinations
def brute_force_analysis(triplet_collection, duo_collection):
    """
    Fixed first triplet for parallel processing
    """
    preset_multiplier = [4, 6, 4, 1]
    cumulative_fan = defaultdict(default_zero)
    hashed_combination_dict = defaultdict(default_zero)
    for first_triplet in triplet_collection:
        for second_triplet in triplet_collection:
            for third_triplet in triplet_collection:
                if test_validity([first_triplet, second_triplet, third_triplet]):
                    for fourth_triplet in triplet_collection:
                        if test_validity(
                            [
                                first_triplet,
                                second_triplet,
                                third_triplet,
                                fourth_triplet,
                            ]
                        ):
                            for duo in duo_collection:
                                if test_validity(
                                    [
                                        first_triplet,
                                        second_triplet,
                                        third_triplet,
                                        fourth_triplet,
                                        duo,
                                    ]
                                ):
                                    tmp_hand = defaultdict(default_zero)
                                    # calculate possibilities of forming such hand
                                    formation_multiplier = 1
                                    win_tile = None
                                    tmp_pack = []
                                    tmp_hash_holder = defaultdict(default_zero)
                                    for k in first_triplet:
                                        if first_triplet[k] == 4:
                                            tmp_pack.append(first_triplet)
                                        else:
                                            tmp_hand[k] += first_triplet[k]
                                        tmp_hash_holder[k] += first_triplet[k]
                                    for k in second_triplet:
                                        if second_triplet[k] == 4:
                                            tmp_pack.append(second_triplet)
                                        else:
                                            tmp_hand[k] += second_triplet[k]
                                        tmp_hash_holder[k] += second_triplet[k]
                                    for k in third_triplet:
                                        if third_triplet[k] == 4:
                                            tmp_pack.append(third_triplet)
                                        else:
                                            tmp_hand[k] += third_triplet[k]
                                        tmp_hash_holder[k] += third_triplet[k]
                                    for k in fourth_triplet:
                                        if fourth_triplet[k] == 4:
                                            tmp_pack.append(fourth_triplet)
                                        else:
                                            tmp_hand[k] += fourth_triplet[k]
                                        tmp_hash_holder[k] += fourth_triplet[k]
                                    for k in duo:
                                        tmp_hand[k] += duo[k]
                                        tmp_hash_holder[k] += duo[k]
                                        win_tile = k
                                    for k in tmp_hand:
                                        formation_multiplier *= preset_multiplier[
                                            tmp_hand[k] - 1
                                        ]
                                    tmp_hand_hash = rule.hash_custom_tiles(
                                        tmp_hash_holder
                                    )
                                    if tmp_hand_hash in hashed_combination_dict.keys():
                                        continue
                                    else:
                                        hashed_combination_dict[
                                            tmp_hand_hash
                                        ] = formation_multiplier
                                    (
                                        fan_sum,
                                        fan_list,
                                    ) = rule.calc_exact_fan_with_PyMahJongGB(
                                        tmp_pack,
                                        tmp_hand,
                                        win_tile,
                                        False,
                                        False,
                                        False,
                                        False,
                                        1,
                                        0,
                                    )
                                    # 嵌张 (1番)：46.71%， 门前清：13.80%
                                    rand_menqing = np.random.randint(0, 10000)
                                    rand_kanzhang = np.random.randint(0, 10000)
                                    # 默认门前清，但根据概率剔除不是门前清的情况
                                    if rand_menqing > 1380:
                                        # 不是门前清时，番值-2
                                        fan_sum -= 2
                                    # 默认单钓将，但根据概率剔除不是单钓将的情况
                                    if rand_kanzhang > 4671:
                                        fan_sum -= 1
                                    if fan_sum >= 8:
                                        for fan in fan_list:
                                            cumulative_fan[fan] += formation_multiplier
    return cumulative_fan, hashed_combination_dict


if __name__ == "__main__":
    triplet_collection = []
    duo_collection = []
    for i in range(len(tile_list_raw)):
        tile = tile_list_raw[i]
        tile_type = tile[0]
        rank = tile[1]
        tile_next = None
        tile_prev = None
        if i != 0:
            tile_prev = tile_list_raw[i - 1]
        if i != len(tile_list_raw) - 1:
            tile_next = tile_list_raw[i + 1]
        # add duo
        duo_collection.append({tile: 2})
        # add trio
        triplet_collection.append({tile: 3})
        triplet_collection.append({tile: 4})
        # add straight
        if (
            tile_next
            and tile_prev
            and tile_next[0] == tile_type
            and tile_prev[0] == tile_type
            and tile_type in ["B", "W", "T"]
        ):
            triplet_collection.append({tile_prev: 1, tile: 1, tile_next: 1})

    fan_dict, hash_dict = brute_force_analysis(triplet_collection, duo_collection)
    # brute_force_filter(triplet_collection[0], triplet_collection, duo_collection)
    # brute_force_analysis(triplet_collection[0], triplet_collection, duo_collection)
    with open("./fan_analysis_simulated.npy", "wb") as f:
        np.save(f, fan_dict)
        np.save(f, hash_dict)
    with open("./fan_analysis_slim.npy", "wb") as f:
        np.save(f, fan_dict)
