#coding:utf-8
import os
import json
import csv
import numpy as np
import math
import argparse
from utils import estimate, qvec2rotmat, np2dic, getGT, getAffineRes

parser = argparse.ArgumentParser()
parser.add_argument("--methodType", required=True)
parser.add_argument("--scene", required=True)
args = parser.parse_args()


def testError(world_coord, new_b, scene):
    x_error = np.sum(np.abs(world_coord[:,0]-new_b[:,0]))
    y_error = np.sum(np.abs(world_coord[:,1]-new_b[:,1]))
    z_error = np.sum(np.abs(world_coord[:,2]-new_b[:,2]))

    num = world_coord.shape[0]
    DRMS = np.sqrt(np.sum(np.square(new_b[:, 0] - world_coord[:, 0]) + np.square(new_b[:, 1] - world_coord[:, 1])) / num)
    DHerr = 20 * np.log10((100 / (DRMS + 0.001)))
    EV = np.sum(np.abs(new_b[:, 2] - world_coord[:, 2])) / num
    DVerr = 25 * np.log10(10 / (EV + 0.001))

    with open(score_path,'a') as f:
        f.write('scene'+str(scene)+' score:\n')
        f.write('x error : '+str(x_error)+'\n')
        f.write('y error : '+str(y_error)+'\n')
        f.write('z error : '+str(z_error)+'\n')
        f.write('DHerr : ' + str(DHerr)+'\n')
        f.write('DVerr : ' + str(DVerr)+'\n\n')


def save_error(fit_error, scene):
    with open(error_path,'a') as f:
        f.write('scene'+str(scene)+' error: '+str(fit_error)+'\n')


def main(gt_path, save_path, data_path, scene):

    gtDict = getGT(gt_path)  # 此处gt_path为scene1.csv， 把第一行去掉了！（remove the first line of scene1.csv）
    t = list()
    with open(data_path,'r') as f:
        [t.append(line) for line in f.read().splitlines()]
    t = t[4:]  # 从colmap中读取生成数据 (read data from colmap image.txt)

    assert len(t)%2==0

    results = list()  # for train data
    world_coord = list()
    testIMG = list()
    testNAME = list()
    trainNAME = list()
    for i in range(int(len(t)/2)):
        c = t[i*2].split(' ')  # colmap remove the no use line, and the data is split by ' '
        quat = np.array((float(c[1]),float(c[2]),float(c[3]),float(c[4])))
        rot_matrix = qvec2rotmat(quat)
        r = rot_matrix[:3,:3]
        T = np.array([[float(c[5]),float(c[6]),float(c[7])]]).T

        x,y,z = -r.T.dot(T)

        if c[-1][:-6] in list(gtDict.keys()): # eg: c[-1] = PIC_20190522_120738_5.jpg
            results.append([x[0],y[0],z[0]])
            world_coord.append(gtDict[c[-1][:-6]])
            trainNAME.append(c[-1][:-6])
        else:
            testIMG.append([x[0],y[0],z[0]])
            testNAME.append(c[-1][:-6])

    for i in list(gtDict.keys()):
        if i not in trainNAME:
            print('gt {} not in data'.format(i))

    clound_coord = np.array(results)   # results for train data
    world_coord = np.array(world_coord)  # ground truth
    testIMG = np.array(testIMG)  # results for test data

    traindic = np2dic(trainNAME, clound_coord)  # merge the six results of one camera to one
    gtdic = np2dic(trainNAME, world_coord)  # merge the six results of one camera to one

    for i,p in enumerate(traindic):
        if i==0:
            trainR = traindic[p]['r']
            gtR = gtdic[p]['r']
        else:
            trainR = np.concatenate([trainR,traindic[p]['r']],0)
            gtR = np.concatenate([gtR, gtdic[p]['r']], 0)
    trainR=trainR.reshape((-1,3))
    gtR=gtR.reshape((-1,3))

    M, train2gt, fit_error = estimate(trainR,gtR)
    save_error(fit_error, scene)
    testError(gtR, train2gt, scene)


    testdic = np2dic(testNAME, testIMG)
    with open(save_path,'w') as f:
        dicname = []
        for i,p in enumerate(testdic):
            dicname.append(p)
            if i == 0:
                testR = testdic[p]['r']
            else:
                testR = np.concatenate([testR, testdic[p]['r']], 0)
        testR = testR.reshape((-1,3))
        testr = getAffineRes(M, testR)
        for i,dn in enumerate(dicname):
            f.write(dn+',%.4f'%testr[i][0]+',%.4f'%testr[i][1]+',%.4f'%testr[i][2]+'\n')

global score_path
global error_path

if args.methodType not in ['org', 'fishEyeMask']:
    raise KeyError('key error')
else:
    score_path = './upload/'+args.methodType+'/score.txt'
    error_path = './upload/'+args.methodType+'/error.txt'

if __name__=='__main__':
    scene = [int(i) for i in args.scene.split(',')]
    for s in scene:
        save_path = './upload/'+args.methodType+'/results'+str(s)+'.txt'
        gt_path = './dataset/gt/scene'+str(s)+'.csv'
        data_path = './workspace/'+args.methodType+'/scene'+str(s)+'/sparse/images.txt'
        main(gt_path, save_path ,data_path ,s)

    import datetime
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n',nowTime)
