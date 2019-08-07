#! /usr/bin/bash

for i in 1
do
PANO=./dataset/pano/scene${i}
SPLIT=./dataset/panoSplit/scene${i}

cd /openMVG/openmvg_build/Linux-x86_64-RELEASE
./openMVG_sample_pano_converter -i $PANO -o $SPLIT -r 1200 -n 6

done
