import os
import sys

sys.path.append(os.path.join(os.getcwd(), ".."))
sys.path.append(os.getcwd())
import src_py.feature
import src_py.rule
import numpy as np
from multiprocessing import Pool
from collections import defaultdict


def default_zero():
    return 0


def postprocessing_profiling(path, file):
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
    p8 = ["妙手回春", "海底捞月", "杠上开花", "抢杠和"]
    p6 = ["全求人"]
    p4 = ["和绝张"]
    fan_list = fan_list.tolist()
    for fan in p8:
        if fan in fan_list:
            fan_sum -= 8
            if fan_sum < 8:
                return fan
    for fan in p6:
        if fan in fan_list:
            fan_sum -= 6
            if fan_sum < 8:
                return fan
    for fan in p4:
        if fan in fan_list:
            fan_sum -= 4
            if fan_sum < 8:
                return fan
    return None


if __name__ == "__main__":
    special_hu_count = defaultdict(default_zero)
    cpuCount = os.cpu_count() - 2
    path = "data"
    # file = "6990.npy"
    # postprocessing_profiling(path, file)
    dir_list = os.listdir(path)

    dir_list.sort()
    pool = Pool(cpuCount)
    ret_list = []
    for fil in dir_list:
        # for fil in file_unprocessed:
        ret = pool.apply_async(
            postprocessing_profiling,
            args=(path, fil),
        )
        ret_list.append(ret)
    pool.close()
    pool.join()
    for ret in ret_list:
        ret = ret.get()
        if ret != None:
            special_hu_count[ret] += 1
    print(special_hu_count)
    print("Total Processed Round: {}".format(len(dir_list)))
