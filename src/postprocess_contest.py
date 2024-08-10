import json
import os
import numpy as np
from fanCalcLib import formMinComb_c
from multiprocessing import Pool
import rule


def load_from_result(dst, file):
    dst_path = os.path.join(dst, file)
    with open(dst_path, "rb") as f:
        init_dist = np.load(f, allow_pickle=True)
        first_shanten = np.load(f, allow_pickle=True)
        first_hu = np.load(f, allow_pickle=True)

        # meta
        winner_id = np.load(f, allow_pickle=True)[0]
        scores = np.load(f, allow_pickle=True)
    return (
        init_dist,  # 1
        first_shanten,  # 2
        first_hu,  # 3
        winner_id,  # 4
        scores,  # 5
    )


def contestprocessing(dst, match_summary, init_tile_list):
    _single_tilewall_length = 34
    scores = match_summary["scores"]
    winner_id = np.argmax(np.array(scores))
    # winner_one_hot = [1 if score > 0 else 0 for score in scores]
    init_wall = json.loads(match_summary["initdata"])["walltiles"].split()
    wind = json.loads(match_summary["initdata"])["quan"]
    game_name = match_summary["_id"]
    file_name = game_name + ".npy"
    dst_path = os.path.join(dst, file_name)
    init_dist = []
    first_shanten = [99, 99, 99, 99]  # shanten using self tiles
    first_hu = [99, 99, 99, 99]  # hu using self tiles
    for i in range(4):
        hand = init_wall[34 * (i + 1) - 1 : 34 * (i + 1) - 14 : -1]
        hand_enc = rule.from_canonical_to_custom_encoding(hand)
        tile_list_cp = rule.update_tile_info(init_tile_list, hand_enc)
        (
            list1,
            list1id,
            list2,
            list2id,
            list3,
            list3id,
            list4,
            list4id,
        ) = formMinComb_c(hand_enc, [], tile_list_cp, i, wind, 15, 7)
        list_comp = []
        if list1id != -1:
            list_comp.append(list1[list1id])
        if list2id != -1:
            list_comp.append(list2[list2id])
        if list3id != -1:
            list_comp.append(list3[list3id])
        if list4id != -1:
            list_comp.append(list4[list4id])
        min_dist = 9
        for entry in list_comp:
            if entry[0] < min_dist:
                min_dist = entry[0]
        min_dist -= 1  # adjust from dist to hu to dist to 上听
        init_dist.append(min_dist)

        if i == 0:
            ending_pos = None
        else:
            ending_pos = _single_tilewall_length * (i) - 1
        hand = init_wall[_single_tilewall_length * (i + 1) - 1 : ending_pos : -1]

        # first shanten and first hu
        got_hu_info_flag = False
        for idx in range(14, _single_tilewall_length):
            if not got_hu_info_flag:
                encountered_tiles = hand[:idx]
                encountered_tiles_custom_encoding = (
                    rule.from_canonical_to_custom_encoding(encountered_tiles)
                )
                tile_list_cp = rule.update_tile_info(
                    tile_list, encountered_tiles_custom_encoding
                )
                (
                    list1,
                    list1id,
                    list2,
                    list2id,
                    list3,
                    list3id,
                    list4,
                    list4id,
                ) = formMinComb_c(
                    encountered_tiles_custom_encoding, [], tile_list_cp, i, wind, 15, 7
                )
                min_selfdraw_dist = 10
                if list1id != -1:
                    selfdraw_dist = list1[list1id][0]
                    if selfdraw_dist < min_selfdraw_dist:
                        min_selfdraw_dist = selfdraw_dist
                if list2id != -1:
                    selfdraw_dist = list2[list2id][0]
                    if selfdraw_dist < min_selfdraw_dist:
                        min_selfdraw_dist = selfdraw_dist
                if list3id != -1:
                    selfdraw_dist = list3[list3id][0]
                    if selfdraw_dist < min_selfdraw_dist:
                        min_selfdraw_dist = selfdraw_dist
                if list4id != -1:
                    selfdraw_dist = list4[list4id][0]
                    if selfdraw_dist < min_selfdraw_dist:
                        min_selfdraw_dist = selfdraw_dist

                if first_shanten[i] > idx and min_selfdraw_dist == 1:
                    first_shanten[i] = idx
                if first_hu[i] > idx and min_selfdraw_dist == 0:
                    first_hu[i] = idx
                    if first_shanten[i] > first_hu[i]:
                        first_shanten[i] = idx
                    got_hu_info_flag = True
    with open(dst_path, "wb") as f:

        np.save(f, np.array(init_dist))
        np.save(f, np.array(first_shanten))
        np.save(f, np.array(first_hu))

        # meta
        np.save(f, [winner_id])
        np.save(f, np.array(scores))
    return


if __name__ == "__main__":
    tile_list = {}

    # fmt: off
    tile_list_raw = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9',  #饼
                'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9',   #万
                'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9',   #条
                'F1', 'F2', 'F3', 'F4', 'J1', 'J2', 'J3' #风、箭
    ]
    # fmt: on

    # 国标综合牌型, 自动搜索 + 算番库
    # 国标特殊牌型(8番以上)，特例搜索

    for tile in tile_list_raw:
        # tile_type = tile[0]
        # tile_rank = int(tile[1])
        tile_list[tile] = 4
    cpuCount = os.cpu_count() - 2
    path_to_contest = "data_src"
    raw_contest_file_list = [
        "gamecontest16e.json",
        "gamecontest16f.json",
        "gamecontest16g.json",
        "gamecontest16h.json",
    ]
    dst = "processed_trial_strict"
    if not os.path.isdir(dst):
        os.makedirs(dst)
    matches_in_json_list = []
    for raw_contest_file in raw_contest_file_list:
        path_to_contest_file = os.path.join(path_to_contest, raw_contest_file)
        with open(path_to_contest_file) as f:
            raw_log = json.load(f)
        matches_in_json_list = matches_in_json_list + raw_log["contest"]["matches"]
    pool = Pool(cpuCount)
    for match_in_json in matches_in_json_list:
        # for fil in file_unprocessed:
        pool.apply_async(
            contestprocessing,
            args=(dst, match_in_json, tile_list),
        )
    pool.close()
    pool.join()
    print("Conversion Processing Done!")
