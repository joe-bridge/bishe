import os
import glob
import numpy as np

anno_path_train = "./data/dataset_dark/voc_norm_train_1391.txt"
anno_path_test = "./data/dataset_dark/voc_norm_test_1391.txt"
img_count = 2782
if os.path.exists(anno_path_train):
    os.remove(anno_path_train)
if os.path.exists(anno_path_test):
    os.remove(anno_path_test)
det_path = "./data/MOT20-02/det/det.txt"
seq_dets = np.loadtxt(det_path, delimiter=',')
img_path = "/home/jiahao/Downloads/Image-Adaptive-YOLO/data/MOT20-02/img1/"
with open(anno_path_train, 'a') as f:
    for frame in range(int(seq_dets[:, 0].max())):
        frame += 1  # detection and frame numbers begin at 1
        if frame > (img_count/2):
            break
        f.write(img_path + str(frame).zfill(6) + '.jpg ')
        dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        for target in range(dets[:, 0].size):
            f.write(str(dets[target, 0]) + ',' + str(dets[target, 1]) + ',' + str(dets[target, 2]) + ','
                    + str(dets[target, 3]) + ',9 ')
        f.write('\n')

with open(anno_path_test, 'a') as f:
    for frame in range(int(img_count/2), int(seq_dets[:, 0].max())):
        frame += 1  # detection and frame numbers begin at 1
        f.write(img_path + str(frame).zfill(6) + '.jpg ')
        dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        for target in range(dets[:, 0].size):
            f.write(str(dets[target, 0]) + ',' + str(dets[target, 1]) + ',' + str(dets[target, 2]) + ','
                    + str(dets[target, 3]) + ',9 ')
        f.write('\n')
