import json
import urllib.request
import os
from multiprocessing import Pool
from fanCalcLib import formMinComb_c
import rule
import numpy as np


def load_from_result(dst, file):
    dst_path = os.path.join(dst, file)
    with open(dst_path, "rb") as f:
        init_dist = np.load(f, allow_pickle=True)
        first_hu = np.load(f, allow_pickle=True)
        first_shanten = np.load(f, allow_pickle=True)
        scores = np.load(f, allow_pickle=True)
    return (
        init_dist,  # 1
        first_hu,  # 2
        first_shanten,  # 3
        scores,  # 4
    )


def check_url(url, dst):
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
    first_shanten = [99, 99, 99, 99]  # shanten using self tiles
    first_hu = [99, 99, 99, 99]  # hu using self tiles
    compose_url = "https://botzone.org.cn/match/{}?lite=true".format(url)
    dst_path = os.path.join(dst, "{}.npy".format(url))
    response = urllib.request.urlopen(compose_url)
    js = json.loads(response.read().decode())
    final_score = js["logs"][-1]["output"]["display"]["score"]
    init_data = json.loads(js["initdata"])
    parsed_init_list = init_data["walltiles"].split(" ")
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
    init_dist = []
    for i in range(4):
        hand = parsed_init_list[34 * (i + 1) - 1 : 34 * (i + 1) - 14 : -1]
        hand_enc = rule.from_canonical_to_custom_encoding(hand)
        tile_list_cp = rule.update_tile_info(tile_list, hand_enc)
        (
            list1,
            list1id,
            list2,
            list2id,
            list3,
            list3id,
            list4,
            list4id,
        ) = formMinComb_c(hand_enc, [], tile_list_cp, i, init_data["quan"], 15, 7)
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

    for id in range(4):
        # first shanten and first hu
        got_hu_info_flag = False
        for i in range(14, _single_tilewall_length):
            if not got_hu_info_flag:
                encountered_tiles = init_tilewall_list[id][:i]
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
                    encountered_tiles_custom_encoding,
                    [],
                    tile_list_cp,
                    id,
                    init_data["quan"],
                    15,
                    7,
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
    # print("Printing stats for game id {}".format(url))
    # print(init_dist)
    # print(first_hu)
    # print(first_shanten)
    with open(dst_path, "wb") as f:

        np.save(f, np.array(init_dist))
        np.save(f, np.array(first_hu))
        np.save(f, np.array(first_shanten))
        np.save(f, np.array(final_score))
    return


if __name__ == "__main__":
    dst = "human_data"
    if not os.path.isdir(dst):
        os.makedirs(dst)
    cpuCount = os.cpu_count() - 2
    url_list = "61fcd0eab8ebe3727badf20c,61fcd252b8ebe3727badf306,61fcd4a5b8ebe3727badf3de,61fcd68db8ebe3727badf560,61fcd8c3b8ebe3727badf6d8,61fcda6ab8ebe3727badf7b5,61fcdc0fb8ebe3727badf8cb,61fcdd56b8ebe3727badf992,61fcdfe3b8ebe3727badfa81,61fce122b8ebe3727badfb54,61fce27ab8ebe3727badfc13,61fce41bb8ebe3727badfcac,61fce57db8ebe3727badfdd8,61fce661b8ebe3727badfe5c,61fce806b8ebe3727badff22,61fcea71b8ebe3727bae009b,62dcee63244d3605b244bec7,62dcef40244d3605b244bf65,62dcf0b0244d3605b244c080,62dcf209244d3605b244c17d,62dcf518244d3605b244c413,62dcf635244d3605b244c4bb,62dcf6b5244d3605b244c541,62dcf79c244d3605b244c588,62dcf946244d3605b244c703,62dcfae4244d3605b244c876,62dcfc9b244d3605b244c9ca,62dcfd7f244d3605b244cade,62dcfeae244d3605b244cbc0,62dd0045244d3605b244ccd7,62dd067f244d3605b244d1c1,62dd08f5244d3605b244d386"
    url_l = url_list.split(",")
    # url = "62e4debab155124b890fadab"
    # check_url(url, dst)
    pool = Pool(cpuCount)
    for url in url_l:
        pool.apply_async(
            check_url,
            args=(url, dst),
        )
    pool.close()
    pool.join()
    print("Post Processing Done!")
