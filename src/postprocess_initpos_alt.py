import os
from turtle import pos
import feature
import generic
import rule
import numpy as np
from multiprocessing import Pool
import json
import postprocess_general

# postprocess_initpos is specifically dedicated to the question that:
# What would happen, if adding completing round mechanism after the first winner emerge
# Rule: only player after drawing-tile player can hu, i.e. player 1 wins, player 2 draws, only player 2 and 3 are allowed to win.


def load_from_result(dst, file):
    dst_path = os.path.join(dst, file)
    with open(dst_path, "rb") as f:
        winner_id = np.load(f, allow_pickle=True)[0]
        winner_fan = np.load(f, allow_pickle=True)[0]
        completing_winner_ids = np.load(f, allow_pickle=True)
        completing_winner_id_fans = np.load(f, allow_pickle=True)
    return (
        winner_id,
        winner_fan,
        completing_winner_ids,
        completing_winner_id_fans,
    )


def postprocessing_initpos(path, dst, processed_src, bz_raw, file):
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

    (
        init_dist,  # 1
        play_count,  # 2
        play_to_shanten,  # 3
        winner_id_alt,  # 4
        fan_list_alt,  # 5
    ) = postprocess_general.load_from_result(processed_src, file)

    raw_log_filename = bz_id + ".json"
    path_to_log = os.path.join(bz_raw, raw_log_filename)
    with open(path_to_log) as f:
        raw_log = json.load(f)
    init_log = json.loads(raw_log["initdata"])
    parsed_init_list = init_log["walltiles"].split(" ")
    init_tilewall_list = []  # Holder for init_tile_wall
    _single_tilewall_length = 34
    for i in range(4):
        lower_bound = None
        if _single_tilewall_length * (i) - 1 > 0:
            lower_bound = _single_tilewall_length * (i) - 1
        init_tilewall_list.append(
            parsed_init_list[_single_tilewall_length * (i + 1) - 1 : lower_bound : -1]
        )

    win_count_by_pos = [[], [], [], []]
    win_point_by_pos = [[], [], [], []]
    mod_win_count_by_pos = [[], [], [], []]
    mod_win_point_by_pos = [[], [], [], []]

    holder_for_completing_round = {}
    completing_round_winner = []
    completing_round_winner_point = []

    # Idea: completing round so that last pos finish the round
    if winner_id != -1 and min(init_dist) != 0:
        # find final stat
        for id in range(winner_id + 1, 4):

            bz_log_init_pointer = 13
            for entry in botzone_log:
                entry_split = entry.split()
                if (
                    len(entry_split) > 2
                    and int(entry_split[1]) == id
                    and entry_split[2] == "Draw"
                ):
                    # print(
                    #     "Player {} draw {}, corresponding to log's tile {}".format(
                    #         id, entry_split[3], init_tilewall_list[id][bz_log_init_pointer]
                    #     )
                    # )
                    bz_log_init_pointer += 1
            # append next tile to consideration list
            if bz_log_init_pointer <= 33:
                tile_considered = init_tilewall_list[id][bz_log_init_pointer]
                for valid_id in range(id, 4):
                    if valid_id not in completing_round_winner:
                        final_hand_tmp = handWall[-1][valid_id]
                        final_wall_tmp = tileWall[-1][valid_id]
                        final_pack_tmp = pack[-1][valid_id]
                        final_hand_list = final_hand_tmp.copy()
                        final_hand_list[tile_considered] += 1
                        final_hand_list["shown"] = final_pack_tmp
                        final_wall_list = final_wall_tmp.copy()
                        final_wall_list[tile_considered] -= 1
                        result = rule.check_hu(
                            final_hand_list,
                            final_wall_list,
                            tile_considered,
                            valid_id,
                            wind,
                            valid_id == id,
                            False,
                        )
                        if result >= 8 and id not in completing_round_winner:
                            completing_round_winner.append(valid_id)
                            completing_round_winner_point.append(result)

    with open(dst_path, "wb") as f:

        np.save(f, np.array([winner_id], dtype=object))
        np.save(f, np.array([fan_sum], dtype=object))
        np.save(f, np.array(completing_round_winner, dtype=object))
        np.save(f, np.array(completing_round_winner_point, dtype=object))


if __name__ == "__main__":
    cpuCount = os.cpu_count() - 1
    path = "data"
    dst = "processed_initpos_alt"
    bz_src = "bz_log_raw"
    processed_src = "processed"
    dir_list = os.listdir(path)
    # file = "10373.npy"
    # postprocessing_initpos(path, dst, processed_src, bz_src, file)

    # dir_list = os.listdir(dst)
    # src_list = os.listdir(path)
    # unprocessed_list = []
    # for fil in src_list:
    #     if fil not in dir_list:
    #         unprocessed_list.append(fil)
    # dir_list = unprocessed_list
    pool = Pool(cpuCount)
    for fil in dir_list:
        # for fil in file_unprocessed:
        pool.apply_async(
            postprocessing_initpos,
            args=(path, dst, processed_src, bz_src, fil),
        )
    pool.close()
    pool.join()
    print("Post Processing Done!")
