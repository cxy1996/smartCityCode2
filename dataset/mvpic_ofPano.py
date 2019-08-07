import os
import sys

bpath = './dataset/officialData'
npath = './dataset/pano'
datasets = os.listdir(bpath)

for dataset in datasets:
    if dataset[:5] != 'scene':
        continue

    scene = dataset.split('_')[0]

    if not os.path.exists(os.path.join(npath, scene)):
        os.mkdir(os.path.join(npath, scene))
    imgs = os.listdir(os.path.join(bpath,dataset))

    for img in imgs:
        if img[:3] != 'PIC':
            continue

        ipath = os.path.join(bpath,dataset,img,'thumbnail.jpg')
        topath = os.path.join(npath,scene,img+'.jpg')
        os.system('cp '+ipath+' '+topath)