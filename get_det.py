import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config_lowlight import cfg
from core.yolov3_lowlight import YOLOV3
from core.config_lowlight import args
import random
import time
exp_folder = os.path.join(args.exp_dir, 'exp_{}'.format(args.exp_num))
TF_ENABLE_DEPRECATION_WARNINGS = 1


if args.use_gpu == 0:
    gpu_id = '-1'
else:
    gpu_id = args.gpu_id
    gpu_list = list()
    gpu_ids = gpu_id.split(',')
    for i in range(len(gpu_ids)):
        gpu_list.append('/gpu:%d' % int(i))
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id


class YoloTest(object):
    def __init__(self):
        self.input_size       = cfg.TEST.INPUT_SIZE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes      = len(self.classes)
        self.anchors          = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.score_threshold  = cfg.TEST.SCORE_THRESHOLD
        self.iou_threshold    = cfg.TEST.IOU_THRESHOLD
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.annotation_path  = cfg.TEST.ANNOT_PATH
        self.weight_file      = cfg.TEST.WEIGHT_FILE
        self.write_image      = cfg.TEST.WRITE_IMAGE
        self.write_image_path = cfg.TEST.WRITE_IMAGE_PATH
        self.show_label       = cfg.TEST.SHOW_LABEL
        self.isp_flag = cfg.YOLO.ISP_FLAG

        with tf.name_scope('input'):
            self.input_data = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='input_data')
            self.trainable  = tf.compat.v1.placeholder(dtype=tf.bool,    name='trainable')
            self.input_data_clean   = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='input_data')

        model = YOLOV3(self.input_data, self.trainable, self.input_data_clean)
        self.pred_sbbox, self.pred_mbbox, self.pred_lbbox, self.image_isped, self.isp_params = \
            model.pred_sbbox, model.pred_mbbox, model.pred_lbbox, model.image_isped,model.filter_params

        with tf.name_scope('ema'):
            ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay)
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)

        # self.sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.saver = tf.compat.v1.train.Saver(ema_obj.variables_to_restore())
        self.saver.restore(self.sess, self.weight_file)

    def predict(self, image, image_name):
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape
        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox, image_isped, isp_param = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox, self.image_isped, self.isp_params],
            feed_dict={
                self.input_data: image_data,  # image_data*np.exp(lowlight_param*np.log(2)),
                self.trainable: False
            }
        )

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)
        if self.isp_flag:
            print('ISP params :  ', isp_param)
            image_isped = np.clip(image_isped[0, ...]*255, 0, 255)
            image_isped = utils.image_unpreporcess(image_isped, [org_h, org_w])
            cv2.imwrite(self.write_image_path + image_name, image_isped)
        else:
            image_isped = np.clip(image, 0, 255)
            # image_isped = utils.image_unpreporcess(image_isped, [org_h, org_w])
            # cv2.imwrite(self.write_image_path + 'low'+ image_name, image_isped)

        return bboxes

    def get_det(self):
        if os.path.exists("./experiments_lowlight/exp_58/track/det.txt"):
            os.remove("./experiments_lowlight/exp_58/track/det.txt")
        path_list = os.listdir("data/MOT20-02/img1")
        path_list.sort()
        frame_num = 1
        for img in path_list:
            image = cv2.imread(os.path.join("data/MOT20-02/img1", img))
            info = self.predict(image, img)
            with open("./experiments_lowlight/exp_58/track/det.txt", mode='a') as f:
                for single_info in info:
                    f.write(str(frame_num) + ",-1," + str(single_info[0]) + ',' + str(single_info[1]) + ','
                            + str(single_info[2]-single_info[0]) + ',' + str(single_info[3]-single_info[1]) + ','
                            + str(single_info[4]) + ",-1,-1,-1" + '\n')
            frame_num = frame_num + 1


if __name__ == '__main__': YoloTest().get_det()