import os
import sys

sys.path.append(os.path.join(os.getcwd(), ".."))
sys.path.append(os.getcwd())
from xml.dom.expatbuilder import parseString
import src_py.feature
import src_py.rule
import numpy as np
from multiprocessing import Pool
from fanCalcLib import formMinComb_c
import json

# Postprocess_selfdraw is for 起始最佳作牌方向, 最终作牌方向, 还有牌墙自摸胡、上听
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


def postprocessing_selfdraw(path, dst, bz_raw, file):
    # fmt: off
    tile_list_raw = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9',  #饼
                'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9',   #万
                'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9',   #条
                'F1', 'F2', 'F3', 'F4', 'J1', 'J2', 'J3' #风、箭


    ]
    # fmt: on
    dst_path = os.path.join(dst, file)
    (
        botzone_log,
        tileWall,
        pack,
        handWall,
        obsWall,
        remaining_tile,
        bz_id,
        winner_id,
        wind,
        fan_sum,
        score,
        fan_list,
    ) = src_py.feature.load_log(path, file)

    tile_list = {}
    for tile in tile_list_raw:
        tile_list[tile] = 4

    init_min_dist = []  # list of dist
    init_hand_hash = [[], [], [], []]  # list of hashes, representing initial hand
    # list of hashes, representing initial targets that have min dist from init hand
    init_min_dist_hash = [[], [], [], []]
    final_min_dist = []
    # list of hashes, representing final targets that have min dist from final hand
    final_min_dist_hash = [[], [], [], []]
    # how many tiles are kept from beginning to end
    count_of_tiles_from_init = []  # list of values
    count_of_tiles_from_others = []  # list of tiles from other players
    first_shanten = [99, 99, 99, 99]  # shanten using self tiles
    first_hu = [99, 99, 99, 99]  # hu using self tiles

    raw_log_filename = bz_id + ".json"
    path_to_log = os.path.join(bz_raw, raw_log_filename)
    with open(path_to_log) as f:
        raw_log = json.load(f)
    init_log = json.loads(raw_log["initdata"])
    parsed_init_list = init_log["walltiles"].split(" ")
    init_tilewall_list = []
    _single_tilewall_length = 34
    for i in range(4):
        if i == 0:
            ending_pos = None
        else:
            ending_pos = _single_tilewall_length * (i) - 1
        init_tilewall_list.append(
            parsed_init_list[_single_tilewall_length * (i + 1) - 1 : ending_pos : -1]
        )
    # find final distance
    for id in range(4):
        # for each player

        # first shanten and first hu
        got_hu_info_flag = False
        for i in range(14, _single_tilewall_length):
            if not got_hu_info_flag:
                encountered_tiles = init_tilewall_list[id][:i]
                encountered_tiles_custom_encoding = (
                    src_py.rule.from_canonical_to_custom_encoding(encountered_tiles)
                )
                tile_list_cp = src_py.rule.update_tile_info(
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
                    encountered_tiles_custom_encoding, [], tile_list_cp, id, wind, 15, 7
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

                if first_shanten[id] > i and min_selfdraw_dist == 1:
                    first_shanten[id] = i
                if first_hu[id] > i and min_selfdraw_dist == 0:
                    first_hu[id] = i
                    if first_shanten[id] > first_hu[id]:
                        first_shanten[id] = i
                    got_hu_info_flag = True

        for i in range(len(botzone_log) - 1, max(len(botzone_log) - 30, 4), -1):
            # find first hands on
            bz_log = botzone_log[i].split()
            if (
                len(bz_log) > 1
                and int(bz_log[1]) == id
                and bz_log[2] in ["Draw", "Peng", "Chi", "Hu"]
            ):
                final_hand_tmp = handWall[i][id]
                final_wall_tmp = tileWall[i][id]
                final_pack_tmp = pack[i][id]
                break

        # calculate final distance
        # find final distance with rule
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
            final_hand_tmp, final_pack_tmp, final_wall_tmp, id, wind, 15, 7
        )
        list_comp = []
        if list1id != -1:
            list_comp.append(list1[list1id])
        if list2id != -1:
            list_comp.append(list2[list2id])
        if list3id != -1:
            list_comp.append(list3[list3id])
        if list4id != -1:
            list_comp.append(list4[list4id])
        if (list4id + list3id + list2id + list1id) == -4:
            final_min_dist.append(9)
            final_min_dist_hash[id].append("")
        else:
            min_dist = 9
            min_dist_ele = None
            for list_ele in list_comp:
                if min_dist > list_ele[0]:
                    min_dist = list_ele[0]
                    min_dist_ele = list_ele
            min_dist -= 1  # from dist2hu to dist2shanten
            final_min_dist.append(min_dist)
            tmp_comb = list(final_pack_tmp) + min_dist_ele[4]
            final_min_dist_hash[id].append(
                src_py.rule.hash_seperated_custom_tile(tmp_comb)
            )

        # find initial distance with rule

        tokens = botzone_log[id + 1].split()
        init_hand_list = tokens[3:]
        customEncHandDict = src_py.rule.from_canonical_to_custom_encoding(
            init_hand_list
        )
        init_hand_hash[id].append(src_py.rule.hash_custom_tiles(customEncHandDict))
        tile_list_cp = src_py.rule.update_tile_info(tile_list, customEncHandDict)
        (
            list1,
            list1id,
            list2,
            list2id,
            list3,
            list3id,
            list4,
            list4id,
        ) = formMinComb_c(customEncHandDict, [], tile_list_cp, id, wind, 15, 7)
        list_comp = []
        if list1id != -1:
            list1.append(list1id)
            list_comp.append(list1)
        if list2id != -1:
            list2.append(list2id)
            list_comp.append(list2)
        if list3id != -1:
            list3.append(list3id)
            list_comp.append(list3)
        if list4id != -1:
            list4.append(list4id)
            list_comp.append(list4)

        # find consistent tiles from beginning to end
        canonical_init_hand = init_hand_list

        # form final hand
        final_hand_list = final_hand_tmp
        final_hand_list["shown"] = final_pack_tmp

        # find tiles sourcing from others
        if isinstance(final_hand_list["shown"], list):
            pack_length = len(final_hand_list["shown"])
            count_of_tiles_from_others.append(pack_length)
        else:
            count_of_tiles_from_others.append(0)

        if (
            isinstance(final_hand_list["shown"], list)
            and len(final_hand_list["shown"]) > 0
        ):
            for entry in final_hand_list["shown"]:
                for k in entry:
                    final_hand_list[k] += entry[k]
        del final_hand_list["shown"]
        canonical_final_hand = src_py.rule.from_custom_to_canonical_encoding(
            final_hand_list
        )
        consistent_tile_count = 0
        for tile in canonical_final_hand:
            if tile in canonical_init_hand:
                consistent_tile_count += 1
                canonical_init_hand.remove(tile)

        count_of_tiles_from_init.append(consistent_tile_count)

        min_dist = 9
        for list_ele in list_comp:
            min_id = list_ele[-1]
            if min_dist > list_ele[min_id][0]:
                min_dist = list_ele[min_id][0]
        min_dist -= 1  # from dist2hu to dist2shanten
        init_min_dist.append(min_dist)

        for list_ele in list_comp:

            for entry in list_ele:
                if isinstance(entry, list) or isinstance(entry, tuple):
                    if entry[0] == min_dist + 1:
                        init_min_dist_hash[id].append(
                            src_py.rule.hash_seperated_custom_tile(entry[4])
                        )
    with open(dst_path, "wb") as f:

        np.save(f, np.array(init_min_dist))
        np.save(f, np.array(init_hand_hash, dtype=object))
        np.save(f, np.array(init_min_dist_hash, dtype=object))
        np.save(f, np.array(final_min_dist_hash, dtype=object))
        np.save(f, np.array(count_of_tiles_from_init))
        np.save(f, np.array(count_of_tiles_from_others))

        np.save(f, np.array(first_shanten))
        np.save(f, np.array(first_hu))
        np.save(f, np.array([fan_sum]))
        np.save(f, np.array(score))
        np.save(f, np.array([winner_id]))


if __name__ == "__main__":
    cpuCount = os.cpu_count() - 2
    path = "data"
    dst = "processed_enhanced"
    bz_src = "bz_log_raw"
    if not os.path.isdir(dst):
        os.makedirs(dst)
    # file = "6990.npy"
    # postprocessing_selfdraw(path, dst, bz_src, file)
    dir_list = os.listdir(path)

    file_list = os.listdir(dst)
    unprocessed_list = []
    for file in dir_list:
        if file not in file_list:
            unprocessed_list.append(file)
    dir_list = unprocessed_list
    print(len(dir_list))

    dir_list.sort()
    pool = Pool(cpuCount)
    for fil in dir_list:
        # for fil in file_unprocessed:
        pool.apply_async(
            postprocessing_selfdraw,
            args=(path, dst, bz_src, fil),
        )
    pool.close()
    pool.join()
    print("Post Processing Done!")
