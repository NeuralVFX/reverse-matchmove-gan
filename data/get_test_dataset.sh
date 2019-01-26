#!/bin/bash


URL=http://neuralvfx.com/datasets/reverse_matchmove/chiang_mai_hi.rar
ZIP_FILE=./data/chiang_mai_hi.rar
TARGET_DIR=./data/
wget -N $URL -O $ZIP_FILE
unrar x $ZIP_FILE  $TARGET_DIR