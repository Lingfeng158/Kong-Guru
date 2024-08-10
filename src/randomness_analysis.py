from multiprocessing import Pool
import os
import numpy as np
import postprocess_general_loader
import postprocess_contest_loader

# Analyzing randomness, with init-dist


def load_from_result(dst, file):
    dst_path = os.path.join(dst, file)
    with open(dst_path, "rb") as f:
        player_win_count_list = np.load(f, allow_pickle=True)
        player_score_list = np.load(f, allow_pickle=True)
        valid_list_length = np.load(f, allow_pickle=True)[0]
        total_list_length = np.load(f, allow_pickle=True)[0]
    return (
        player_win_count_list,
        player_score_list,
        valid_list_length,
        total_list_length,
    )


def trial(valid_list, dst, seed):
    np.random.seed(seed)
    load_from_general = False
    if dst == "processed/":
        load_from_general = True
    #     loader_function = postprocess_general_loader.load_from_result
    # else:
    #     loader_function = postprocess_contest_loader.load_from_result
    # dir_list = os.listdir(dst)
    max_cumulative_length = 2300
    player_win_count = [[0, 0, 0, 0]]
    player_score = [[0, 0, 0, 0]]

    for i in range(max_cumulative_length):
        if len(player_score) >= 2000 and len(player_win_count) >= 2000:
            break
        file_id = np.random.randint(0, len(valid_list))
        file = valid_list[file_id]
        if load_from_general:
            (
                init_dist,
                play_count,
                play_to_shanten,
                winner_id,
                fan_sum,
                score,
                fan_list,
            ) = postprocess_general_loader.load_from_result(dst, file)
            one_hot_win = [0, 0, 0, 0]
            if winner_id >= 0 and winner_id < 4:
                one_hot_win[(winner_id + i) % 4] = 1
                score_offset = [score[(j - i) % 4] for j in range(4)]
                cum_win_count = [
                    a + b for a, b in zip(one_hot_win, player_win_count[-1])
                ]
                cum_score = [a + b for a, b in zip(score_offset, player_score[-1])]
                player_win_count.append(cum_win_count)
                player_score.append(cum_score)
        else:
            (
                init_dist,  # 1
                first_shanten,  # 2
                first_hu,  # 3
                winner_id,  # 4
                score,  # 5
            ) = postprocess_contest_loader.load_from_result(dst, file)
            one_hot_win = [0, 0, 0, 0]
            if winner_id >= 0 and winner_id < 4:
                one_hot_win[(winner_id + i) % 4] = 1
                score_offset = [score[(j + i) % 4] for j in range(4)]
                cum_win_count = [
                    a + b for a, b in zip(one_hot_win, player_win_count[-1])
                ]
                cum_score = [a + b for a, b in zip(score_offset, player_score[-1])]
                player_win_count.append(cum_win_count)
                player_score.append(cum_score)

    return player_win_count, player_score


def experiment(valid_list, src_path, name_on_save):
    cpuCount = os.cpu_count() - 2
    pool = Pool(cpuCount)
    player_win_count_list = []
    player_score_list = []
    ret_list = []
    valid_list_length = len(valid_list)
    dir_list_length = len(os.listdir(src_path))
    for i in range(1000):
        # for fil in file_unprocessed:
        ret = pool.apply_async(
            trial,
            args=(valid_list, src_path, i),
        )
        ret_list.append(ret)
    pool.close()
    pool.join()
    for ret in ret_list:

        player_win_count, player_score = ret.get()
        player_win_count_list.append(player_win_count)
        player_score_list.append(player_score)

    with open(name_on_save, "wb") as f:
        np.save(f, np.array(player_win_count_list, dtype=object))
        np.save(f, np.array(player_score_list, dtype=object))
        np.save(f, np.array([valid_list_length], dtype=object))
        np.save(f, np.array([dir_list_length], dtype=object))


if __name__ == "__main__":
    # Analysis on Random Init

    valid_list_strict_3 = []
    valid_list_strict_4 = []
    valid_list_above_2_diff_1 = []
    valid_list_above_2 = []

    dst_trial = "processed_trial_strict/"
    dir_trial_list = os.listdir(dst_trial)
    for i in range(len(dir_trial_list)):
        file = dir_trial_list[i]
        (
            init_dist,  # 1
            first_shanten,  # 2
            first_hu,  # 3
            winner_id,  # 4
            scores,  # 5
        ) = postprocess_contest_loader.load_from_result(dst_trial, file)
        if list(init_dist) == [3, 3, 3, 3]:
            valid_list_strict_3.append(file)
        if list(init_dist) == [4, 4, 4, 4]:
            valid_list_strict_4.append(file)

    #     if min(init_dist) >= 3:
    #         valid_list_above_2.append(file)
    #         if max(init_dist) - min(init_dist) <= 1:
    #             valid_list_above_2_diff_1.append(file)

    dst_general = "processed/"
    dir_general_list = os.listdir(dst_general)
    # for i in range(len(dir_general_list)):
    #     file = dir_general_list[i]
    #     (
    #         init_dist,
    #         play_count,
    #         play_to_shanten,
    #         winner_id,
    #         fan_sum,
    #         score,
    #         fan_list,
    #     ) = postprocess_general_loader.load_from_result(dst_general, file)
    #     if min(init_dist) >= 3:
    #         valid_list_above_2.append(file)
    #         if max(init_dist) - min(init_dist) <= 1:
    #             valid_list_above_2_diff_1.append(file)

    config = [
        # [valid_list_strict_4, dst_trial, "dat_strict_4.npy"],
        [
            valid_list_strict_4 + valid_list_strict_3,
            dst_trial,
            "dat_strict_3_or_4_slim.npy",
        ],
        # [valid_list_above_2_diff_1, dst_general, "dat_above_2_diff_1.npy"],
        # [valid_list_above_2, dst_general, "dat_above_2.npy"],
        # [dir_general_list, dst_general, "dat_no_restriction_slim.npy"],
    ]

    # start experiments
    for i in config:
        l, src, name = i
        experiment(l, src, name)
