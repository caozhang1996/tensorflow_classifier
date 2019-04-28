#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from preprocessing import inception_preprocessing
from preprocessing import vgg_preprocessing

preprocessing_fn_map = {'inception': inception_preprocessing,
                        'inception_v1': inception_preprocessing,
                        'inception_v2': inception_preprocessing,
                        'inception_v3': inception_preprocessing,
                        'inception_v4': inception_preprocessing,
                        'inception_resnet_v2': inception_preprocessing,
                        'mobilenet_v1': inception_preprocessing,
                        'mobilenet_v2': inception_preprocessing,
                        'mobilenet_v2_035': inception_preprocessing,
                        'mobilenet_v2_140': inception_preprocessing,
                        'nasnet_mobile': inception_preprocessing,
                        'nasnet_large': inception_preprocessing,
                        'pnasnet_mobile': inception_preprocessing,
                        'pnasnet_large': inception_preprocessing,
                        'resnet_v1_50': vgg_preprocessing,
                        'resnet_v1_101': vgg_preprocessing,
                        'resnet_v1_152': vgg_preprocessing,
                        'resnet_v1_200': vgg_preprocessing,
                        'resnet_v2_50': vgg_preprocessing,
                        'resnet_v2_101': vgg_preprocessing,
                        'resnet_v2_152': vgg_preprocessing,
                        'resnet_v2_200': vgg_preprocessing,
                        'vgg': vgg_preprocessing,
                        'vgg_a': vgg_preprocessing,
                        'vgg_16': vgg_preprocessing,
                        'vgg_19': vgg_preprocessing,
                          }

def get_preprocessing(name, is_training=False):
    """Returns preprocessing_fn(image, height, width, **kwargs).
    Args:
        name: The name of the preprocessing function.
        is_training: `True` if the model is being used for training and `False`
        otherwise.
    Returns:
        preprocessing_fn: A function that preprocessing a single image (pre-batch).
        It has the following signature:
        image = preprocessing_fn(image, output_height, output_width, ...).
    Raises:
        ValueError: If Preprocessing `name` is not recognized.
    """

    if name not in preprocessing_fn_map:
        raise ValueError('Preprocessing name [%s] was not recognized' % name)

    def preprocessing_fn(image, output_height, output_width, **kwargs):
        return preprocessing_fn_map[name].preprocess_image(image,
                                                           output_height,
                                                           output_width,
                                                           is_training=is_training,
                                                           **kwargs)

    return preprocessing_fn
