import os
import sys

orgin_path = './dataset/officialData'
to_path = './dataset/fishEye'
dir = os.listdir(orgin_path)

for i,dataset in enumerate(dir):
    if dataset[0:5] != 'scene':
        continue

    bpath = os.path.join(orgin_path,dir[i])
    picdir = os.listdir(bpath)
    scene = dataset.split('_')[0]
    for pic in picdir:
        if pic[:3] != 'PIC':
            continue

        img = os.path.join(bpath,pic)
        all = os.listdir(img)

        scene_path = os.path.join(to_path, scene)
        tpath = os.path.join(scene_path,pic)

        if not os.path.exists(scene_path):
            os.mkdir(scene_path)
        for ith, fish in enumerate(all):
            if fish[0:6]=='origin':
                fpath = os.path.join(img,fish)
                final_tpath = tpath+'_'+str(int(fish[7])-1)+'.jpg'
                os.system('cp ' + fpath + ' ' + final_tpath)