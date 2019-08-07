#coding:utf-8
import os
import json
import csv
import numpy as np
import math


def qvec2angle(qvec):
    w = float(qvec[0])
    x = float(qvec[1])
    y = float(qvec[2])
    z = float(qvec[3])

    r = math.atan((2 * (w * x + y * z)) / (1 - 2 * (x * x + y * y)))
    p = math.asin(2 * (w * y - x * z))
    y = math.atan((2 * (w * z + x * y)) / (1 - 2 * (z * z + y * y)))

    def norm(x):
        if x < 0:
            return 360 + x
        else:
            return x

    angleR = r * 180 / math.pi
    angleP = p * 180 / math.pi
    angleY = y * 180 / math.pi
    return [norm(int(angleR)), norm(int(angleP)), norm(int(angleY))]

def affine_fit(from_pts, to_pts):
    q = from_pts
    p = to_pts
    assert len(q) == len(p)
    assert len(q) > 1
    dim = len(q[0])
    assert len(q) > dim

    c = [[0.0 for a in range(dim)] for i in range(dim + 1)]
    for j in range(dim):
        for k in range(dim + 1):
            for i in range(len(q)):
                qt = list(q[i]) + [1]
                c[k][j] += qt[k] * p[i][j]

    Q = [[0.0 for a in range(dim)] + [0] for i in range(dim + 1)]
    for qi in q:
        qt = list(qi) + [1]
        for i in range(dim + 1):
            for j in range(dim + 1):
                Q[i][j] += qt[i] * qt[j]

    M = [Q[i] + c[i] for i in range(dim + 1)]

    if not valid(M):
        return False

    class transformation:
        def transform(self, pt):
            res = [0.0 for a in range(dim)]
            for j in range(dim):
                for i in range(dim):
                    res[j] += pt[i] * M[i][j + dim + 1]
                res[j] += M[dim][j + dim + 1]
            return res
    return transformation(), M

def estimate(from_pt, to_pt):
    trn, M = affine_fit(from_pt, to_pt)
    result = []
    err = 0
    for i in range(len(from_pt)):
        fp = from_pt[i]
        tp = to_pt[i]
        t = trn.transform(fp)
        result.append(t)
        err += ((tp[0] - t[0]) ** 2 + (tp[1] - t[1]) ** 2) ** 0.5
    print("fit error = %f" % err)
    return M, np.array(result), err

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def np2dic(testNAME, coord):
    testdic = dict()
    for i,name in enumerate(testNAME):
        if name not in list(testdic.keys()):
            testdic[name]=dict()
            testdic[name]['num']=np.copy(coord[i])
            testdic[name]['k']=1
        else:
            testdic[name]['num']+=np.copy(coord[i])
            testdic[name]['k']+=1

    for n in list(testdic.keys()):
        testdic[n]['r']=testdic[n]['num']/testdic[n]['k']

    return testdic

def getGT(gt_path):
    csvData = []
    try:
        # csvFile = csv.reader(open(gt_path, 'r'))
        csvFile = csv.reader(open(gt_path, 'r', encoding='utf-8'))
        for scene in csvFile:
            csvData.append(scene)
    except:
        csvFile = csv.reader(open(gt_path, 'r', encoding='gbk'))
        for scene in csvFile:
            csvData.append(scene)
    csvData = csvData[1:]
    gtDict = {}
    for data in csvData:
        if data[0] not in gtDict.keys():
            gtDict[data[0]] = [float(data[1]),
                               float(data[2]),
                               float(data[3])]
    return gtDict

def getAffineRes(M, testR):
    resdim = testR[0].shape[0]
    testr = []
    for d in range(testR.shape[0]):
        pt = testR[d]
        res = [0 for i in range(resdim)]
        for j in range(resdim):
            for i in range(resdim):
                res[j] += pt[i] * M[i][j + resdim + 1]
            res[j] += M[resdim][j + resdim + 1]
        testr.append(res)
    testr = np.array(testr)
    return testr

def valid(m, eps=1.0 / (10 ** 10)):
    (h, w) = (len(m), len(m[0]))
    for y in range(0, h):
        maxrow = y
        for y2 in range(y + 1, h):
            if abs(m[y2][y]) > abs(m[maxrow][y]):
                maxrow = y2
        (m[y], m[maxrow]) = (m[maxrow], m[y])
        if abs(m[y][y]) <= eps:
            return False
        for y2 in range(y + 1, h):
            c = m[y2][y] / m[y][y]
            for x in range(y, w):
                m[y2][x] -= m[y][x] * c
    for y in range(h - 1, 0 - 1, -1):
        c = m[y][y]
        for y2 in range(0, y):
            for x in range(w - 1, y - 1, -1):
                m[y2][x] -= m[y][x] * m[y2][y] / c
        m[y][y] /= c
        for x in range(h, w):
            m[y][x] /= c
    return True