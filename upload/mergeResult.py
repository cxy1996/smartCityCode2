import os
import numpy as np

topath1 = './upload/fishEyeMask/results1.txt'
topath2 = './upload/fishEyeMask/results6.txt'
sort = True
use_example = False
final_res = [topath1, topath2]

with open('./upload.txt','w') as f:
    for file in final_res:
        with open(file, 'r') as g:
            [f.write(line+'\n') for line in g.read().splitlines()]


# sort
if sort:
    with open('./upload.txt', 'r') as f:
        datas = [line.split(',') for line in f.read().splitlines()]

    if use_example:
        with open('./upload/b.txt', 'r') as f:
            ids = [line.split(',')[0] for line in f.read().splitlines()]
    else:
        test1 = os.listdir('./dataset/officialData/scene1_jiading_lib_test')
        test2 = os.listdir('./dataset/officialData/scene6_jiading_bolou_test')
        ids = []
        ids.extend(test1)
        ids.extend(test2)

    result = dict()
    for d in datas:
        result[d[0]] = d[1:]

    a = list()
    for t in ids:
        if t in result.keys():
            for i in result[t]:
               t = t+','+i
            a.append(t)
        else:
            a.append(t)

    with open('sortResult.txt','w') as f:
        for i in a:
            f.write(i+'\n')