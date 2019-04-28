#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import sys
import numpy as np
import cv2
import os
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim

# tfrecord文件个数
_NUM_SHARDS = 5

# 验证数据的个数
_NUM_VALIDATION = 350

# seed for random
_RANDOM_SEED = 0

_RESIZE_HEIGHT = 224
_RESIZE_WIDTH = 224

LABELS_FILENAME = 'labels.txt'
SPLIT_TO_SIZES = {'train': 3320, 'validation': 350}
_ITEMS_TO_DESCRIPTIONS = {'image': 'A color image of varying size.',
                          'label': 'A single integer between 0 and 4',
                          }
_NUM_CLASSES = 5


# %%
# 生成整数型的属性
def int64_feature(value):
    if not isinstance(value, (tuple, list)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# 生成字符串型的属性
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 生成实数型的属性
def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def image_to_tfexample(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature={'image/encoded': bytes_feature(image_data),
                                                                'image/format': bytes_feature(image_format),
                                                                'image/class/label': int64_feature(class_id),
                                                                'image/height': int64_feature(height),
                                                                'image/width': int64_feature(width),
                                                                }
                                                       ))

# %% 创建一个图片读取类
class ImageReader():

    def __init__(self):
        self._decode_jpeg_data = tf.placeholder(tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def decode_jpeg(self, sess, data):
        image = sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def read_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]


# %% 另一个读取图片的函数
def read_image(filename, resize_height, resize_width, normalization=False):
    """
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename: 图片名
    :param resize_height:
    :param resize_width:
    :param normalization: 是否归一化到[0.,1.0]
    :return:
    """
    bgr_image = cv2.imread(filename)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)    # 将bgr图像转换成rgb图像

    if resize_width > 0 and resize_height > 0:
        rgb_image = cv2.resize(rgb_image, (resize_width, resize_height))
    rgb_image = np.asanyarray(rgb_image)

    if normalization:
        rgb_image = rgb_image / 255.0
    return rgb_image


def get_filenames_and_classes(dataset_dir):
    """

    :param dataset_dir: 数据集根目录
    :return:
    """
    directories = []
    class_names = []
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    image_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            image_filenames.append(path)

    return image_filenames, sorted(class_names)


def get_tfrecord_filename(dataset_dir, split_name, shard_id):
    """

    :param dataset_dir:  数据集根目录
    :param split_name:  'train' or 'validation'
    :param shard_id: tfrecord文件id
    :return:
    """
    output_filename = 'flower_224_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


def tfrecord_exists(dataset_dir):
    """
    判断tfrecord文件是否存在,有一个不存在就返回False
    :param dataset_dir:
    :return:
    """
    for split_name in ['train', 'validation']:
        for shard_id in range(_NUM_SHARDS):
            output_filename = get_tfrecord_filename(dataset_dir, split_name, shard_id)
            if not tf.gfile.Exists(output_filename):
                return False
    return True


# %% 下面的代码都是与labels.txt文件有关的
def write_label_file(labels_to_class_names, dataset_dir, filename=LABELS_FILENAME):
    """Writes a file with the list of class names.
    Args:
        labels_to_class_names: A map of (integer) labels to class names.
        dataset_dir: The directory in which the labels file should be written.
        filename: The filename where the class names are written.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))


def has_labels(dataset_dir, filename=LABELS_FILENAME):
    """Specifies whether or not the dataset directory contains a label map file.
    Args:
        dataset_dir: The directory in which the labels file is found.
        filename: The filename where the class names are written.
    Returns:
        `True` if the labels file exists and `False` otherwise.
    """
    return tf.gfile.Exists(os.path.join(dataset_dir, filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
    """Reads the labels file and returns a mapping from ID to class name.
     Args:
        dataset_dir: The directory in which the labels file is found.
        filename: The filename where the class names are written.
    Returns:
        A map from a label (integer) to class name.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'rb') as f:
        lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index+1:]
    return labels_to_class_names


# %%
def create_tfrecord(split_name, filenames, class_names_to_ids, dataset_dir):
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

    image_reader = ImageReader()
    with tf.Session() as sess:
        for shard_id in range(_NUM_SHARDS):
            output_filename = get_tfrecord_filename(dataset_dir, split_name, shard_id)

            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                # 开始索引和结束索引
                start_ndx = shard_id * num_per_shard
                end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))

                for i in range(start_ndx, end_ndx):
                    sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i + 1, len(filenames), shard_id))
                    sys.stdout.flush()

                    # Read the filename:
                    image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                    height, width = image_reader.read_dims(sess, image_data)

                    # os.path.basename('D:\CSDN')返回CSDN
                    # os.path.dirname()
                    class_name = os.path.basename(os.path.dirname(filenames[i]))
                    class_id = class_names_to_ids[class_name]

                    example = image_to_tfexample(image_data,
                                                 b'jpg',
                                                 height,
                                                 width,
                                                 class_id)
                    tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


# %%
def read_tfrecords(split_name, dataset_dir, file_pattern=None, reader=None):
    """

    :param split_name: train or validation
    :param dataset_dir: 存放数据的根目录
    :param file_pattern: 文件格式，型如: flower_224_train_* or flower_224_validation_*
    :param reader:
    :return:
    """
    if split_name not in {'train': 3320, 'validation': 350}:
        raise ValueError('split name %s was not recognized.' % split_name)

    if file_pattern is None:
        file_pattern = 'flower_224_%s_*.tfrecord'
    # such as './flower_224_train_*-of-*.tfrecord' or './flower_224_validation_*-of-*.tfrecord'
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
                        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
                        'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
                        }

    items_to_handlers = {'image': slim.tfexample_decoder.Image(),
                         'label': slim.tfexample_decoder.Tensor('image/class/label'),
                         }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    labels_to_names = None
    if has_labels(dataset_dir):
        labels_to_names = read_label_file(dataset_dir)

    return slim.dataset.Dataset(data_sources=file_pattern,
                                reader=reader,
                                decoder=decoder,
                                num_samples=SPLIT_TO_SIZES[split_name],
                                items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
                                num_classes=_NUM_CLASSES,
                                labels_to_names=labels_to_names)


# %%
def run(dataset_dir):
    """Runs the download and conversion operation.
    Args:
        dataset_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    if not tfrecord_exists(dataset_dir):
        print('Tfrecord files are not exist. please create them.')
        return

    image_filenames, class_names = get_filenames_and_classes(dataset_dir)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))  # 将类别名称转化为id, class_names_to_id是一个字典

    # Divide into train and test:
    random.seed(_RANDOM_SEED)
    random.shuffle(image_filenames)
    training_filenames = image_filenames[_NUM_VALIDATION:]       # 将数据分成train和validation
    validation_filenames = image_filenames[:_NUM_VALIDATION]

    # First, convert the training and validation sets.
    create_tfrecord('train', training_filenames, class_names_to_ids, dataset_dir)
    create_tfrecord('validation', validation_filenames, class_names_to_ids, dataset_dir)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    write_label_file(labels_to_class_names, dataset_dir)

    print('\nFinished converting the Flowers dataset!')


if __name__ == '__main__':
    run('/home/caozhang/tensorflow_project/dataset/flower_photos/')

