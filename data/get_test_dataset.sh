#!/bin/bash


URL=http://neuralvfx.com/datasets/reverse_matchmove/chiang_mai.rar
ZIP_FILE=./data/chiang_mai.rar
TARGET_DIR=./data/
wget -N $URL -O $ZIP_FILE
unrar x $ZIP_FILE  $TARGET_DIR