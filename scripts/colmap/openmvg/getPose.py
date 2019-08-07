import json
import csv
import numpy
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from affineTrans import test,affine_fit

def quaternion_matrix(quaternion):
    # Return homogeneous rotation matrix from quaternion.
    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    n = numpy.dot(q, q)
    if n < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / n)
    q = numpy.outer(q, q)
    return numpy.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

def affine_matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True):
    v0 = numpy.array(v0, dtype=numpy.float64, copy=True)
    v1 = numpy.array(v1, dtype=numpy.float64, copy=True)

    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError("input arrays are of wrong shape or type")

    # move centroids to origin
    t0 = -numpy.mean(v0, axis=1)
    M0 = numpy.identity(ndims+1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -numpy.mean(v1, axis=1)
    M1 = numpy.identity(ndims+1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)

    if shear:
        # Affine transformation
        A = numpy.concatenate((v0, v1), axis=0)
        u, s, vh = numpy.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims:2*ndims]
        t = numpy.dot(C, numpy.linalg.pinv(B))
        t = numpy.concatenate((t, numpy.zeros((ndims, 1))), axis=1)
        M = numpy.vstack((t, ((0.0,)*ndims) + (1.0,)))
    elif usesvd or ndims != 3:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = numpy.linalg.svd(numpy.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = numpy.dot(u, vh)
        if numpy.linalg.det(R) < 0.0:
            # R does not constitute right handed system
            R -= numpy.outer(u[:, ndims-1], vh[ndims-1, :]*2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = numpy.identity(ndims+1)
        M[:ndims, :ndims] = R
    else:
        # Rigid transformation matrix via quaternion
        # compute symmetric matrix N
        xx, yy, zz = numpy.sum(v0 * v1, axis=1)
        xy, yz, zx = numpy.sum(v0 * numpy.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = numpy.sum(v0 * numpy.roll(v1, -2, axis=0), axis=1)
        N = [[xx+yy+zz, 0.0,      0.0,      0.0],
             [yz-zy,    xx-yy-zz, 0.0,      0.0],
             [zx-xz,    xy+yx,    yy-xx-zz, 0.0],
             [xy-yx,    zx+xz,    yz+zy,    zz-xx-yy]]
        # quaternion: eigenvector corresponding to most positive eigenvalue
        w, V = numpy.linalg.eigh(N)
        q = V[:, numpy.argmax(w)]
        q /= vector_norm(q)  # unit quaternion
        # homogeneous transformation matrix
        M = quaternion_matrix(q)

    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= math.sqrt(numpy.sum(v1) / numpy.sum(v0))

    # move centroids back
    M = numpy.dot(numpy.linalg.inv(M1), numpy.dot(M, M0))
    M /= M[ndims, ndims]
    return M

def superimposition_matrix(v0, v1, scale=True, usesvd=True):
    # Return matrix to transform given 3D point set into second point set.
    v0 = numpy.array(v0, dtype=numpy.float64, copy=False)[:3]
    v1 = numpy.array(v1, dtype=numpy.float64, copy=False)[:3]
    return affine_matrix_from_points(v0, v1, shear=False,
                                     scale=scale, usesvd=usesvd)

def align_reconstruction_naive_similarity(X, Xp):
    """Align with GPS and GCP data using direct 3D-3D matches."""
    # Compute similarity Xp = s A X + b

    T = superimposition_matrix(X.T, Xp.T, scale=True)
    A, b = T[:3, :3], T[:3, 3]
    s = np.linalg.det(A)**(1. / 3)
    A /= s
    return s, A, b

def show1(aa,bb,name):
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    for i in range(len(aa)):
        # plot point
        ax.scatter(aa[i][0],aa[i][1],aa[i][2], c='y')
        ax.scatter(bb[i][0],bb[i][1],bb[i][2], c='r')
        # plot line
        x=np.array([aa[i][0],bb[i][0]])
        y=np.array([aa[i][1],bb[i][1]])
        z=np.array([aa[i][2],bb[i][2]])
        ax.plot(x,y,z,c='b')
        # # plot text
        # label = '%s' % (name[i])
        # ax.text(bb[i][0], bb[i][1], bb[i][2], label, color='red')
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

def show(aa):
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    for i in range(len(aa)):
        # plot point
        ax.scatter(aa[i][0],aa[i][1],aa[i][2], c='y')
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

def error_test(world_coord, new_b, scene):
    x_error = np.sum(np.abs(world_coord[:,0]-new_b[:,0]))
    y_error = np.sum(np.abs(world_coord[:,1]-new_b[:,1]))
    z_error = np.sum(np.abs(world_coord[:,2]-new_b[:,2]))

    num = world_coord.shape[0]
    DRMS = np.sqrt(np.sum(np.square(new_b[:, 0] - world_coord[:, 0]) + np.square(new_b[:, 1] - world_coord[:, 1])) / num)
    DHerr = 20 * np.log10((100 / (DRMS + 0.001)))
    EV = np.sqrt(np.sum(np.abs(new_b[:, 2] - world_coord[:, 2])) / num)
    DVerr = 25 * np.log10(10 / (EV + 0.001))

    with open(score_path,'a') as f:
        f.write('scene'+str(scene)+' score:\n')
        f.write('x error : '+str(x_error)+'\n')
        f.write('y error : '+str(y_error)+'\n')
        f.write('z error : '+str(z_error)+'\n')
        f.write('DHerr : ' + str(DHerr)+'\n')
        f.write('DVerr : ' + str(DVerr)+'\n\n')

def affine_res(M, testR):
    resdim = testR[0].shape[0]
    testr = []
    for d in range(testR.shape[0]):
        pt = testR[d]
        res = [0.0 for a in range(resdim)]
        for j in range(resdim):
            for i in range(resdim):
                res[j] += pt[i] * M[i][j + resdim + 1]
            res[j] += M[resdim][j + resdim + 1]
        testr.append(res)
    testr = np.array(testr)
    return testr

def main(sfm_data_path, gt_path, save_path, scene, method):
    with open(sfm_data_path,"r") as f:
        data = {}  # data{img_name: value}
        all_data = json.loads(f.read()) # all reconstruction imgs

        view_key2value = {}
        extr_key2value = {}
        for img in all_data['views']:
            view_key2value[img['key']] = img['value']

        for img in all_data['extrinsics']:
            extr_key2value[img['key']] = img['value']

        # not all extrinsics has [img['key']]
        for key in view_key2value:
            try:
                img_name = view_key2value[key]['ptr_wrapper']['data']['filename']
                data[img_name] = extr_key2value[key]
            except:
                pass

    try:
        csv_file = csv.reader(open(gt_path,'r',encoding='utf-8'))
        csv_data=[]
        for stu in csv_file:
            csv_data.append(stu)
    except:
        csv_file = csv.reader(open(gt_path,'r',encoding='gbk'))
        csv_data=[]
        for stu in csv_file:
            csv_data.append(stu)

    csv_data = csv_data[1:]  # ground truth

    data_train={}
    for gt in csv_data:
        try:
            data_train[gt[0]+'.jpg'] = data[gt[0]+'.jpg']
        except:
            print('gt {} not in data'.format(gt[0]))
            pass

    world_coord=[]
    clound_coord=[]
    name=[]
    for i in range(len(csv_data)):
        if (csv_data[i][0]+'.jpg') in data_train:
            a=np.array(csv_data[i][1:],dtype=np.float)
            b=np.array(data[csv_data[i][0]+'.jpg']['center'],dtype=np.float)
            world_coord.append(a)
            clound_coord.append(b)
            name.append(csv_data[i][0])

    world_coord=np.array(world_coord)
    clound_coord=np.array(clound_coord)

    if method==1:
        s, A, b=align_reconstruction_naive_similarity(clound_coord, world_coord)
        new_b=s*A.dot(clound_coord.T).T+b
        error_test(world_coord, new_b, scene)

    elif method==2:
        M, train2gt = test(clound_coord, world_coord)
        train2gt_v2 = affine_res(M, clound_coord)
        assert train2gt.all() == train2gt_v2.all()
        error_test(world_coord, train2gt, scene)

    #show1(world_coord,new_b,name)

    cc=[]

    data_name = data.keys()
    for i in data_name:
        c =np.array(data[i]['center'],dtype=np.float)
        cc.append(c)

    cc=np.array(cc)

    if method==1:
        new_c=s*A.dot(cc.T).T+b

    elif method==2:
        new_c = affine_res(M, cc)

    #show(new_c)
    train_img = list()
    for img in csv_data:
        train_img.append(img[0])
    with open(save_path,'w') as f:
        for i,img in enumerate(data_name):
            name = img.split('.')[0]
            if name not in train_img:
                f.write(name+',%.4f'%new_c[i][0]+',%.4f'%new_c[i][1]+',%.4f'%new_c[i][2]+'\n')

score_path = './results/score.txt'
if __name__ == '__main__':
    import os
    if os.path.exists(score_path):
        os.system('rm '+score_path)

    #for i in np.arange(1,9):
    for i in [1,2,3,4,5,6,7,8]:
        sfm_data_path = '/home/cxy/smartcityData/results/scene'+str(i)+'/reconstruction/sfm_data.json'
        gt_path = '/home/cxy/smartcityData/all_dataset/gt/scene'+str(i)+'.csv'
        save_path = './upload/results'+str(i)+'.txt'
        main(sfm_data_path, gt_path, save_path, i, method=1)



