#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
import nets.inception_v1 as inception_v1
import tensorflow.contrib.slim as slim
from preprocessing import preprocessing_factory
from datasets import convert_flower_to_tfrecord

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


num_classes = 5
batch_size = 16                # batch_size不宜过大，否者会出现内存不足的问题
resize_height = 224
resize_width = 224
channels = 3

input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, channels])
input_labels = tf.placeholder(dtype=tf.int32, shape=[None, num_classes])

keep_prob = tf.placeholder(dtype=tf.float32)
is_training = tf.placeholder(dtype=tf.bool)


def train(dataset_dir,
          base_lr=0.01,
          max_steps=30000,
          train_log_dir='./logs/inception_v1_Momentum_0.01',
          preprocessing_name='inception_v1'
          ):
    """
    :param dataset_dir: 存放数据的根目录
    :param base_lr: 学习率
    :param max_steps: 最大迭代次数
    :param train_log_dir: 模型文件的存放位置
    :param preprocessing_name: 所使用的预处理名称
    :return:
    """
    dataset = convert_flower_to_tfrecord.read_tfrecords(split_name='train',
                                                        dataset_dir=dataset_dir)

    provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                              num_readers=4,  # The number of parallel readers that read data from the dataset.
                                                              common_queue_capacity=20 * batch_size,
                                                              common_queue_min=10 * batch_size)

    [images, labels] = provider.get(['image', 'label'])

    images_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training=True)
    images = images_preprocessing_fn(images, resize_height, resize_width)


    # 得到batch_size大小的图像和标签
    train_batch_images, train_batch_labels = tf.train.batch([images, labels],
                                                            batch_size=batch_size,
                                                            num_threads=4,   # The number of threads used to create the batches.
                                                            capacity= 5 * batch_size)


    # 对标签进行one-hot编码
    train_batch_labels = tf.one_hot(train_batch_labels, num_classes, on_value=1, off_value=0)


    with slim.arg_scope(inception_v1.inception_v1_arg_scope()):  # inception_v1.inception_v1_arg_scope()括号不能掉，表示一个函数
        out, end_points = inception_v1.inception_v1(inputs=input_images,
                                                    num_classes=num_classes,
                                                    is_training=is_training,
                                                    dropout_keep_prob=keep_prob)

    # 计算loss，accuracy，选择优化器
    loss = tf.losses.softmax_cross_entropy(onehot_labels=input_labels, logits=out)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(input_labels, 1)), tf.float32)) * 100.0

    optimizer = tf.train.MomentumOptimizer(learning_rate=base_lr, momentum=0.9)  # 这里可以使用不同的优化函数

    # 在定义训练的时候, 注意到我们使用了`batch_norm`层时,需要更新每一层的`average`和`variance`参数,
    # 正常的训练过程不包括更新，需要我们去手动像下面这样更新
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):  # 执行完更新操作之后，再进行训练操作
        train_op = slim.learning.create_train_op(total_loss=loss, optimizer=optimizer)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for steps in np.arange(max_steps):
            sess.run([images, labels])
            input_batch_images, input_batch_labels = sess.run([train_batch_images, train_batch_labels])
            _, train_loss = sess.run([train_op, loss], feed_dict={input_images: input_batch_images,
                                                                  input_labels: input_batch_labels,
                                                                  keep_prob: 0.8,
                                                                  is_training: True})
            # 得到训练过程中的loss， accuracy值
            if steps % 50 == 0 or (steps + 1) == max_steps:
                train_acc = sess.run(accuracy, feed_dict={input_images: input_batch_images,
                                                          input_labels: input_batch_labels,
                                                          keep_prob: 1.0,
                                                          is_training: False})
                print('Step: %d, loss: %.4f, accuracy: %.4f' % (steps, train_loss, train_acc))


            # 每隔2000步储存一下模型文件
            if steps % 2000 == 0 or (steps + 1) == max_steps:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=steps)

        coord.request_stop()
        coord.join(threads)


# %%
if __name__ == "__main__":
    train(dataset_dir='/home/caozhang/tensorflow_project/dataset/flower_photos/')
