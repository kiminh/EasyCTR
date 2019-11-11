#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from common import fc
from common import add_hidden_layer_summary
from common import check_arg
from common import get_feature_vectors
from common import project


"""
Product-based Neural Networks for User Response Prediction over Multi-field Categorical Data

The name 'w*' means adding field weight (singe variable) for every pair of interactions.
"""


def _check_wkfm_args(args):
    check_arg(args, 'wkfm_use_shared_embedding', bool)
    check_arg(args, 'wkfm_use_project', bool)
    check_arg(args, 'wkfm_project_size', int)


def get_wkfm_logits(
        features,
        feature_columns,
        shared_feature_vectors,
        units,
        is_training,
        extra_options):

    with tf.variable_scope('wkfm'):
        _check_wkfm_args(extra_options)
        use_shared_embedding = extra_options['wkfm_use_shared_embedding']
        use_project = extra_options['wkfm_use_project']
        project_size = extra_options['wkfm_project_size']

        if not use_shared_embedding:
            feature_vectors = get_feature_vectors(features, feature_columns)
        else:
            feature_vectors = shared_feature_vectors

        if use_project:
            feature_vectors = project(feature_vectors, project_size)

        y = _wkfm(feature_vectors, reduce_sum=True)  # [B, 1]
        with tf.variable_scope('logits') as logits_scope:
            # fc just for adding a bias
            logits = fc(y, units=units, name=logits_scope)
            add_hidden_layer_summary(logits, logits_scope.name)

        return logits


def _wkfm(feature_vectors, reduce_sum=True):
    """Kernel FM
      feature_vectors: List of shape [B, ?] tensors, size N

    Half-Optimized implementation

    Return:
      Tensor of shape [B, T] if reduce_sum is True, or shape [B, 1], T is the sum
      dimentions of all features.
    """

    with tf.variable_scope('wkfm'):
        outputs = []
        x = tf.concat(feature_vectors, axis=1)   # [B, T]
        T = x.shape[1].value
        N = len(feature_vectors)
        indices = []
        for j in range(N):
            vj = feature_vectors[j]
            dj = vj.shape[1].value
            indices.extend([j] * dj)

        for i in range(N):
            vi = feature_vectors[i]
            name = 'wkfm_{}'.format(i)
            di = vi.shape[1].value
            U = tf.get_variable(name, [T, di])

            # 创建两两交叉特征的权重, 与 KFM 的主要区别就是这个权重
            name = 'wkfm_weightes_{}'.format(i)
            wkfm_weights = tf.get_variable(name, [N], initializer=tf.ones_initializer)
            weights = tf.gather(wkfm_weights, indices)

            y = tf.matmul(weights * x, U)   # [B, di]
            outputs.append(y)
        y = tf.concat(outputs, axis=1)   # [B, T]
        y = x * y  # [B, T]

        if reduce_sum:
            y = tf.reduce_sum(y, axis=1, keepdims=True)  # [B, 1]

        return y
