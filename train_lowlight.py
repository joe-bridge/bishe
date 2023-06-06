#! /usr/bin/env python
# coding=utf-8


import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
from core.dataset_lowlight import Dataset
from core.yolov3_lowlight import YOLOV3
from core.config_lowlight import cfg
from core.config_lowlight import args
import random
TF_ENABLE_DEPRECATION_WARNINGS = 1


# 设置GPU编号和使用哪些GPU
if args.use_gpu == 0:
    gpu_id = '-1'
else:
    gpu_id = args.gpu_id
    gpu_list = list()
    gpu_ids = gpu_id.split(',')
    for i in range(len(gpu_ids)):
        gpu_list.append('/gpu:%d' % int(i))
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

# 设置实验目录
exp_folder = os.path.join(args.exp_dir, 'exp_{}'.format(args.exp_num))

# 设置checkpoint保存目录
set_ckpt_dir = args.ckpt_dir
args.ckpt_dir = os.path.join(exp_folder, set_ckpt_dir)
if not os.path.exists(args.ckpt_dir):
    os.makedirs(args.ckpt_dir)

# 写入配置信息到config.txt
config_log = os.path.join(exp_folder, 'config.txt')
arg_dict = args.__dict__
msg = ['{}: {}\n'.format(k, v) for k, v in arg_dict.items()]
utils.write_mes(msg, config_log, mode='w')


class YoloTrain(object):
    def __init__(self):
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.learn_rate_init = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods = cfg.TRAIN.WARMUP_EPOCHS
        self.initial_weight = cfg.TRAIN.INITIAL_WEIGHT
        self.time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.max_bbox_per_scale = 150
        self.train_logdir = "./data/log/train"
        self.trainset = Dataset('train')
        self.testset = Dataset('test')
        self.steps_per_period = len(self.trainset)
        # 开启一个会话
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        # self.sess                = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        # 定义输入
        with tf.name_scope('define_input'):
            self.input_data = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='input_data')
            self.label_sbbox = tf.compat.v1.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox = tf.compat.v1.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox = tf.compat.v1.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes = tf.compat.v1.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes = tf.compat.v1.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.compat.v1.placeholder(dtype=tf.float32, name='lbboxes')
            self.input_data_clean = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='input_data')

            self.trainable = tf.compat.v1.placeholder(dtype=tf.bool, name='training')

        # 定义YOLOv3模型及其损失函数，并计算总的损失
        with tf.name_scope("define_loss"):
            self.model = YOLOV3(self.input_data, self.trainable, self.input_data_clean)
            t_variables = tf.compat.v1.trainable_variables()
            print("t_variables", t_variables)
            # self.net_var = [v for v in t_variables if not 'extract_parameters' in v.name]
            self.net_var = tf.compat.v1.global_variables()
            self.giou_loss, self.conf_loss, self.prob_loss, self.recovery_loss = self.model.compute_loss(
                self.label_sbbox, self.label_mbbox, self.label_lbbox,
                self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
            # self.loss only includes the detection loss.
            self.loss = self.giou_loss + self.conf_loss + self.prob_loss

        # 定义学习率，根据全局步数的大小选择不同的学习率，同时更新全局步数
        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                       dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant((self.first_stage_epochs + self.second_stage_epochs) * self.steps_per_period,
                                      dtype=tf.float64, name='train_steps')
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) * (1 + tf.cos(
                                     (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            global_step_update = tf.compat.v1.assign_add(self.global_step, 1.0)

        # 定义权重衰减和指数加权平均模型，用于正则化和平滑更新模型参数
        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.compat.v1.trainable_variables())
            # tf.trainable_variables() 返回需要被训练的变量，这些变量会在优化过程中进行更新。
            # tf.train.ExponentialMovingAverage() 定义了一个指数加权平均模型，用于平滑模型的变量值。
            # apply() 方法将指数加权平均模型应用到可训练变量上，使它们的值平滑更新，从而达到正则化的目的。

        # 定义第一阶段训练操作，只更新YOLOv3的第一阶段中的卷积层参数
        with tf.name_scope("define_first_stage_train"):
            self.first_stage_trainable_var_list = []
            for var in tf.compat.v1.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:  # 根据变量名的前缀将训练的可训练变量列表筛选出来
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.compat.v1.train.AdamOptimizer(
                self.learn_rate).minimize(self.loss, var_list=self.first_stage_trainable_var_list)
            # 用这些变量创建了一个 Adam 优化器，以最小化定义好的损失函数 self.loss。

            with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
                # 使用 tf.control_dependencies 将优化器与 tf.GraphKeys.UPDATE_OPS 集合中的更新操作和 moving_ave 操作绑定

                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()  # 最后返回一个空操作 tf.no_op() 作为训练操作

        # 定义第二阶段训练操作，更新YOLOv3的所有参数
        with tf.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.compat.v1.trainable_variables()
            second_stage_optimizer = tf.compat.v1.train.AdamOptimizer(
                self.learn_rate).minimize(self.loss, var_list=second_stage_trainable_var_list)

            with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

        # 加载和保存TensorFlow模型中的变量
        with tf.name_scope('loader_and_saver'):
            self.loader = tf.compat.v1.train.Saver(self.net_var)
            # self.loader 是用于加载预训练模型的 Saver 对象，它只需要保存预训练模型中的变量。
            self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=5)
            # self.saver 是用于保存训练过程中所有变量的 Saver 对象，它需要保存所有变量，包括预训练模型中的变量以及在训练过程中新添加的变量。
            # max_to_keep 参数指定最多保留的模型数量。

        # 定义TensorFlow的摘要操作，用于记录模型的训练情况并可视化。
        with tf.name_scope('summary'):
            tf.compat.v1.summary.scalar("learn_rate", self.learn_rate)
            tf.compat.v1.summary.scalar("giou_loss", self.giou_loss)
            tf.compat.v1.summary.scalar("conf_loss", self.conf_loss)
            tf.compat.v1.summary.scalar("prob_loss", self.prob_loss)
            tf.compat.v1.summary.scalar("recovery_loss", self.recovery_loss)
            tf.compat.v1.summary.scalar("total_loss", self.loss)

            # logdir = "./data/log/"
            logdir = os.path.join(exp_folder, 'log')

            if os.path.exists(logdir):
                shutil.rmtree(logdir)
            os.mkdir(logdir)
            self.write_op = tf.compat.v1.summary.merge_all()  # tf.summary.merge_all 将所有摘要合并成一个操作，用于将摘要写入文件
            self.summary_writer = tf.compat.v1.summary.FileWriter(logdir, graph=self.sess.graph)
            # tf.summary.FileWriter 初始化一个摘要写入器，指定了写入的文件夹和图形。

    def train(self):
        self.sess.run(tf.compat.v1.global_variables_initializer())
        try:
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.loader.restore(self.sess, self.initial_weight)
        except ValueError:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train YOLOV3 from scratch ...')
            self.first_stage_epochs = 0

        for epoch in range(1, 1 + self.first_stage_epochs + self.second_stage_epochs):
            # 训练一个物体检测模型，包括两个阶段的训练。
            # 第一阶段训练的是冻结层的权重，第二阶段训练所有的权重。在每个阶段内，使用不同的训练操作 train_op 进行训练。
            if epoch <= self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables

            pbar = tqdm(self.trainset)  # 通过 tqdm 库来显示训练进度条。进度条都是train的，train完了进度条结束再进行test。
            train_epoch_loss, test_epoch_loss = [], []
            # 将训练集和测试集的 loss 值分别存储到 train_epoch_loss 和 test_epoch_loss 列表中

            # 对训练数据
            for train_data in pbar:
                # train_data[0]：训练图像数据
                # train_data[1]-[3]：对应的小、中、大尺度的标签
                # train_data[4]-[6]：对应的小、中、大尺度的真实边界框
                if args.lowlight_FLAG:  # 对训练数据，如果 args.lowlight_FLAG 为真，则需要进行低光照处理，否则不做处理
                    lowlight_param = 1
                    if random.randint(0, 2) > 0:
                        lowlight_param = random.uniform(1.5, 5)  # 使用随机参数 lowlight_param 对图像进行低光照处理
                    _, summary, train_step_loss, train_step_loss_recovery, global_step_val = self.sess.run(
                        [train_op, self.write_op, self.loss, self.recovery_loss, self.global_step], feed_dict={
                            self.input_data: np.power(train_data[0], lowlight_param),  # 将低光处理后的图像作为输入
                            self.label_sbbox: train_data[1],
                            self.label_mbbox: train_data[2],
                            self.label_lbbox: train_data[3],
                            self.true_sbboxes: train_data[4],
                            self.true_mbboxes: train_data[5],
                            self.true_lbboxes: train_data[6],
                            self.input_data_clean: train_data[0],
                            self.trainable: True,
                        })
                else:
                    _, summary, train_step_loss, global_step_val = self.sess.run(
                        [train_op, self.write_op, self.loss, self.global_step], feed_dict={
                            self.input_data: train_data[0],
                            self.label_sbbox: train_data[1],
                            self.label_mbbox: train_data[2],
                            self.label_lbbox: train_data[3],
                            self.true_sbboxes: train_data[4],
                            self.true_mbboxes: train_data[5],
                            self.true_lbboxes: train_data[6],
                            self.input_data_clean: train_data[0],
                            self.trainable: True,
                        })

                train_epoch_loss.append(train_step_loss)  # 将 train_step_loss 加入到 train_epoch_loss 列表中
                self.summary_writer.add_summary(summary, global_step_val)

                pbar.set_description("train loss: %.2f" % train_step_loss)

            # 对测试数据
            if args.lowlight_FLAG:
                for test_data in self.testset:
                    # lowlight_param = random.uniform(-2, 0)
                    lowlight_param = 1
                    if random.randint(0, 2) > 0:
                        lowlight_param = random.uniform(1.5, 5)
                    test_step_loss = self.sess.run(self.loss, feed_dict={
                        self.input_data: np.power(test_data[0], lowlight_param),
                        # test_data[0]*np.exp(lowlight_param*np.log(2)),
                        self.label_sbbox: test_data[1],
                        self.label_mbbox: test_data[2],
                        self.label_lbbox: test_data[3],
                        self.true_sbboxes: test_data[4],
                        self.true_mbboxes: test_data[5],
                        self.true_lbboxes: test_data[6],
                        self.input_data_clean: test_data[0],
                        self.trainable: False,
                    })
                    # 使用当前模型对处理后的测试数据进行预测，得到 test_step_loss，并将其加入到 test_epoch_loss 列表中。
                    test_epoch_loss.append(test_step_loss)
            else:
                for test_data in self.testset:
                    test_step_loss = self.sess.run(self.loss, feed_dict={
                        self.input_data: test_data[0],
                        self.label_sbbox: test_data[1],
                        self.label_mbbox: test_data[2],
                        self.label_lbbox: test_data[3],
                        self.true_sbboxes: test_data[4],
                        self.true_mbboxes: test_data[5],
                        self.true_lbboxes: test_data[6],
                        self.input_data_clean: test_data[0],
                        self.trainable: False,
                    })

                    test_epoch_loss.append(test_step_loss)

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            with open("./experiments_lowlight/exp_58/log/log.txt", mode='a') as f:
                f.write(str(train_epoch_loss) + "," + str(test_epoch_loss) + "\n")
            # 根据 test_epoch_loss 的值构造模型保存路径 ckpt_file
            ckpt_file = args.ckpt_dir + "/yolov3_test_loss=%.4f.ckpt" % test_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            # 输出当前 epoch 的信息，包括当前 epoch 编号、当前时间、平均训练 loss 和平均测试 loss，以及模型保存路径
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                  % (epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
            # 使用 saver 保存模型。
            self.saver.save(self.sess, ckpt_file, global_step=epoch)


if __name__ == '__main__':
    if os.path.exists("./experiments_lowlight/exp_58/log/log.txt"):
        os.remove("./experiments_lowlight/exp_58/log/log.txt")
    YoloTrain().train()
