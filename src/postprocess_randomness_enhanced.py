import os
from turtle import pos
from xml.dom.expatbuilder import parseString
import feature
import generic
import rule
import numpy as np
from multiprocessing import Pool
import json

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
        first_shanten = np.load(f, allow_pickle=True)
        first_hu = np.load(f, allow_pickle=True)
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
        first_shanten,  # 8
        first_hu,  # 9
        winner_id,  # 10
    )


def postprocessing_enhanced(path, dst, bz_raw, file):
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
        fan_list,
    ) = feature.load_log(path, file)

    tile_list = {}
    for tile in tile_list_raw:
        tile_list[tile] = 4

    init_min_dist = []  # list of dist
    init_min_dist_comb = [[], [], [], []]  # list of hashes
    init_list_tier = [[], [], [], []]  # list of hashes

    final_retrotracted_dist = []  # find final target comb from initial hands
    final_min_dist = []
    final_min_dist_comb = [[], [], [], []]
    final_list_tier = [[], [], [], []]  # list of hashes
    # how many tiles are kept from beginning to end
    count_of_tiles_from_init = []  # list of values
    count_of_tiles_from_others = []  # list of tiles from other players
    intersection_related = []
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
        for i in range(14, _single_tilewall_length, 2):
            if not got_hu_info_flag:
                encountered_tiles = init_tilewall_list[id][:i]
                encountered_tiles_custom_encoding = (
                    rule.from_canonical_to_custom_encoding(encountered_tiles)
                )
                encountered_tiles_custom_encoding["shown"] = []
                tile_list_cp = rule.update_tile_info(
                    tile_list, encountered_tiles_custom_encoding
                )
                ret_lists = generic.form_min_combination(
                    encountered_tiles_custom_encoding,
                    tile_list_cp,
                    id,
                    wind,
                    max_dist=1,
                )

                for ret_list in ret_lists:
                    if (not got_hu_info_flag) and len(ret_list) > 1:
                        if first_shanten[id] > i and ret_list[ret_list[-1]][0] == 1:
                            first_shanten[id] = i
                        if first_hu[id] > i and ret_list[ret_list[-1]][0] == 0:
                            first_hu[id] = i
                            if first_shanten[id] > first_hu[id]:
                                first_shanten[id] = i
                            got_hu_info_flag = True
                            break

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
        # form hand_list
        final_hand_list = final_hand_tmp
        final_hand_list["shown"] = final_pack_tmp
        # find initial distance with rule

        (list1, list2, list3, list4) = generic.form_min_combination(
            final_hand_list, final_wall_tmp, id, wind
        )
        list_comp = []
        list_comp.append(list1)
        list_comp.append(list2)
        list_comp.append(list3)
        list_comp.append(list4)
        min_dist = 9
        for i in range(4):
            min_id = list_comp[i][-1]
            if len(list_comp[i]) > 1:
                if min_dist > list_comp[i][min_id][0]:
                    min_dist = list_comp[i][min_id][0]
        final_min_dist.append(min_dist)

        for i in range(4):  # where i is iteration of list_comp
            tier_list = []
            for entry in list_comp[i]:
                if isinstance(entry, list) and entry[0] == min_dist:
                    # get target comp
                    hashed_str = rule.hash_seperated_custom_tile(entry[4])
                    final_min_dist_comb[id].append(hashed_str)
                    tier_list.append(i)
            tier_list = list(set(tier_list))
            final_list_tier[id] += tier_list

        # find initial
        # for each player
        for i in range(5, min(30, len(botzone_log))):
            # find first hands on
            bz_log = botzone_log[i].split()

            if (
                len(bz_log) > 1
                and int(bz_log[1]) == id
                and bz_log[2] in ["Draw", "Peng", "Chi"]
            ):
                init_hand_tmp = handWall[i][id]
                init_wall_tmp = tileWall[i][id]
                init_pack_tmp = pack[i][id]
                break
        # calculate initial distance
        # form hand_list
        init_hand_list = init_hand_tmp
        init_hand_list["shown"] = init_pack_tmp

        # find initial distance with rule

        (list1, list2, list3, list4) = generic.form_min_combination(
            init_hand_list, init_wall_tmp, id, wind, result_threshold=45
        )

        # find consistent tiles from beginning to end

        if (
            isinstance(init_hand_list["shown"], list)
            and len(init_hand_list["shown"]) > 0
        ):
            for entry in init_hand_list["shown"]:
                for k in entry:
                    init_hand_list[k] += entry[k]
        del init_hand_list["shown"]
        canonical_init_hand = rule.from_custom_to_canonical_encoding(init_hand_list)

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
        canonical_final_hand = rule.from_custom_to_canonical_encoding(final_hand_list)
        consistent_tile_count = 0
        for tile in canonical_final_hand:
            if tile in canonical_init_hand:
                consistent_tile_count += 1
                canonical_init_hand.remove(tile)

        count_of_tiles_from_init.append(consistent_tile_count)

        list_comp = []
        list_comp.append(list1)
        list_comp.append(list2)
        list_comp.append(list3)
        list_comp.append(list4)
        min_dist = 9
        retro_list_tmp = []

        for i in range(4):
            for entry in list_comp[i]:
                if isinstance(entry, list):
                    targ_hash = rule.hash_seperated_custom_tile(entry[4])
                    if targ_hash in final_min_dist_comb[id]:
                        retro_list_tmp.append(entry[0])
        retro_init_dist = min(retro_list_tmp) if len(retro_list_tmp) > 0 else 9
        final_retrotracted_dist.append(retro_init_dist)

        for i in range(4):  # where i is iteration of list_comp
            min_id = list_comp[i][-1]
            if len(list_comp[i]) > 1:
                if min_dist > list_comp[i][min_id][0]:
                    min_dist = list_comp[i][min_id][0]
        init_min_dist.append(min_dist)

        for i in range(4):  # where i is iteration of list_comp
            tier_list = []
            for entry in list_comp[i]:
                if isinstance(entry, list) and entry[0] == min_dist:
                    # print(entry)
                    # get target comp
                    hashed_str = rule.hash_seperated_custom_tile(entry[4])
                    init_min_dist_comb[id].append(hashed_str)
                    tier_list.append(i)
            tier_list = list(set(tier_list))
            init_list_tier[id] += tier_list

    # find intersection relation
    for id in range(4):
        included = False
        for ent in final_min_dist_comb[id]:
            if ent in init_min_dist_comb[id]:
                included = True
                break
        intersection_related.append(included)

    with open(dst_path, "wb") as f:

        np.save(f, np.array(init_min_dist, dtype=object))
        np.save(f, np.array(init_list_tier, dtype=object))
        np.save(f, np.array(final_retrotracted_dist, dtype=object))
        np.save(f, np.array(final_list_tier, dtype=object))
        np.save(f, np.array(count_of_tiles_from_init))
        np.save(f, np.array(count_of_tiles_from_others))
        np.save(f, np.array(intersection_related))
        np.save(f, np.array(first_shanten))
        np.save(f, np.array(first_hu))
        np.save(f, np.array([winner_id]))


if __name__ == "__main__":
    cpuCount = os.cpu_count() - 2
    path = "data"
    dst = "processed_enhanced"
    bz_src = "bz_log_raw"
    # file = "12850.npy"
    # postprocessing_enhanced(path, dst, bz_src, file)
    dir_list = os.listdir(path)
    pool = Pool(cpuCount)
    for fil in dir_list:
        # for fil in file_unprocessed:
        pool.apply_async(
            postprocessing_enhanced,
            args=(path, dst, bz_src, fil),
        )
    pool.close()
    pool.join()
    print("Post Processing Done!")
