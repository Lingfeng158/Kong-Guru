import os
from turtle import pos
import feature
import generic
import rule
import numpy as np
from multiprocessing import Pool

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


def postprocessing_alt(path, dst, file):
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
        fan_list,
    ) = feature.load_log(path, file)

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

    # find final distance
    for id in range(4):
        # for each player
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
        np.save(f, np.array([winner_id]))


if __name__ == "__main__":
    cpuCount = os.cpu_count() - 6
    path = "data"
    dst = "processed_alt"
    # file = "12850.npy"
    # postprocessing_alt(path, dst, file)
    dir_list = os.listdir(path)
    pool = Pool(cpuCount)
    for fil in dir_list:
        # for fil in file_unprocessed:
        pool.apply_async(
            postprocessing_alt,
            args=(path, dst, fil),
        )
    pool.close()
    pool.join()
    print("Post Processing Done!")
