import os
import sys

sys.path.append(os.path.join(os.getcwd(), ".."))
sys.path.append(os.getcwd())
import src_py.feature
from fanCalcLib import formMinComb_c
import src_py.rule
import numpy as np
from multiprocessing import Pool

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


def postprocessing(path, dst, init_tile_list, file):
    dst_path = os.path.join(dst, file)
    (
        botzone_log,
        tileWall,
        pack,
        handWall,
        obsWall,
        remaining_tile,
        _,
        winner_id,
        wind,
        fan_sum,
        score,
        fan_list,
    ) = src_py.feature.load_log(path, file)

    init_dist = []  # 初始上听数
    play_count = [0, 0, 0, 0]  # 上手数
    play_to_shanten = [99, 99, 99, 99]
    # final_dist = []  # 最终上听数
    winner_id = winner_id
    fan_list = fan_list

    # find initial distance
    for id in range(4):
        # for each player
        tokens = botzone_log[id + 1].split()
        init_hand_list = tokens[3:]
        if int(tokens[1]) != id or len(init_hand_list) != 13:
            return -1

        customEncHandDict = src_py.rule.from_canonical_to_custom_encoding(
            init_hand_list
        )
        tile_list = src_py.rule.update_tile_info(init_tile_list, customEncHandDict)
        (
            list1,
            list1id,
            list2,
            list2id,
            list3,
            list3id,
            list4,
            list4id,
        ) = formMinComb_c(customEncHandDict, [], tile_list, id, wind, 15, 7)
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
        init_dist.append(max(min_dist, 0))

        # calculate 上手数
        play_count_temp = 0
        is_shanten = False
        for i in range(1, len(botzone_log)):
            bz_log = botzone_log[i].split()

            if (
                len(bz_log) > 2
                and int(bz_log[1]) == id
                and bz_log[2] in ["Play", "BuGang", "AnGang", "Gang"]
            ):
                play_count_temp += 1
                # print(handWall[i][id])
                # print(botzone_log[i])
                hand_list = handWall[i][id]
                pack_list = pack[i][id]
                if is_shanten == False:
                    is_shanten = src_py.rule.PyMajJongGB_shanten(hand_list, pack_list)
                    if is_shanten:
                        play_to_shanten[id] = play_count_temp
                # print(is_shanten, i, id)
        play_count[id] = play_count_temp
        # print(play_count_temp)

    with open(dst_path, "wb") as f:

        np.save(f, np.array(init_dist))
        np.save(f, np.array(play_count))
        np.save(f, np.array(play_to_shanten))

        # meta
        np.save(f, [winner_id])
        np.save(f, [fan_sum])
        np.save(f, score)
        np.save(f, fan_list)


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
    path = "data"
    dst = "processed"
    if not os.path.isdir(dst):
        os.makedirs(dst)
    dir_list = os.listdir(path)
    # file_list1 = os.listdir(path)
    # file_list2 = os.listdir(dst)
    # file_unprocessed = list(set(file_list1) - set(file_list2))
    # postprocessing(path, dst, tile_list, "0.npy")
    pool = Pool(cpuCount)
    for fil in dir_list:
        # for fil in file_unprocessed:
        pool.apply_async(
            postprocessing,
            args=(path, dst, tile_list, fil),
        )
    pool.close()
    pool.join()
    print("Post Processing Done!")
