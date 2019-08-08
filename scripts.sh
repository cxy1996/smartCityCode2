#! /usr/bin/bash

sh ./dataset/mvpic_ofPano.py

sh ./dataset/mvpic_ofFish.py

sh ./scripts/colmap/panoSplit.sh

sh ./scripts/colmap/reconstructOrg.sh

sh ./scripts/colmap/reconstructFish+.sh

sh ./scripts/colmap/reconstructOrg_halfPara

python3 ./upload/mergeResult.py
