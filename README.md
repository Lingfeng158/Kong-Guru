# Kong-Guru
An analytical tool for Official International MahJong game.

# Directory Introduction

## src/
Survey bot and human players' characteristics exhibited in historical gameplays.

## data_src/
compressed datasets.

## data/
Under which are processed MahJong match data, ready to use.

## supervised_learning_v7/
Code for preprocessing data, neural version of framework(as models), and fitting

## bot_framework_slim/
Code for uploading fitted bots to botzone.org.cn

## Download additional data from botzone.org.cn

1. download competition info json, i.e. download.ipynb, to bz_raw (raw botzone log) -> /bz_raw/*.json
2. process raw info to intermediate data format, i.e. proess_botzone_raw.py, -> data.txt
3. process intermediate data format to customized data format, i.e. preprocess.py -> /data/*.npy