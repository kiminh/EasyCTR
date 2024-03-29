#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from common import add_hidden_layer_summary
from common import add_hidden_layers
from common import fc
from common import check_arg
from common import pairwise_kernel_dot
from common import get_activation_fn
from common import get_feature_vectors
from common import project


"""
Product-based Neural Networks for User Response Prediction over Multi-field Categorical Data
"""


def _check_kpnn_args(args):
    check_arg(args, 'kpnn_use_shared_embedding', bool)
    check_arg(args, 'kpnn_use_project', bool)
    check_arg(args, 'kpnn_project_size', int)
    check_arg(args, 'kpnn_hidden_units', list)
    check_arg(args, 'kpnn_activation_fn', str)
    check_arg(args, 'kpnn_dropout', float)
    check_arg(args, 'kpnn_batch_norm', bool)
    check_arg(args, 'kpnn_layer_norm', bool)
    check_arg(args, 'kpnn_use_resnet', bool)
    check_arg(args, 'kpnn_use_densenet', bool)
    check_arg(args, 'kpnn_concat_project', bool)

    check_arg(args, 'leaky_relu_alpha', float)
    check_arg(args, 'swish_beta', float)


def get_kpnn_logits(
        features,
        feature_columns,
        shared_feature_vectors,
        units,
        is_training,
        extra_options):

    with tf.variable_scope('kpnn'):
        _check_kpnn_args(extra_options)
        use_shared_embedding = extra_options['kpnn_use_shared_embedding']
        use_project = extra_options['kpnn_use_project']
        project_size = extra_options['kpnn_project_size']
        hidden_units = extra_options['kpnn_hidden_units']
        activation_fn = extra_options['kpnn_activation_fn']
        dropout = extra_options['kpnn_dropout']
        batch_norm = extra_options['kpnn_batch_norm']
        layer_norm = extra_options['kpnn_layer_norm']
        use_resnet = extra_options['kpnn_use_resnet']
        use_densenet = extra_options['kpnn_use_densenet']
        concat_project = extra_options['kpnn_concat_project']
        leaky_relu_alpha = extra_options['leaky_relu_alpha']
        swish_beta = extra_options['swish_beta']
        activation_fn = get_activation_fn(activation_fn=activation_fn,
                                          leaky_relu_alpha=leaky_relu_alpha,
                                          swish_beta=swish_beta)
        if not use_shared_embedding:
            feature_vectors = get_feature_vectors(features, feature_columns)
        else:
            feature_vectors = shared_feature_vectors

        project_feature_vectors = None
        if use_project:
            project_feature_vectors = project(feature_vectors, project_size)

        y = _kpnn(feature_vectors=feature_vectors,
                  project_feature_vectors=project_feature_vectors,
                  use_project=use_project,
                  units=units,
                  hidden_units=hidden_units,
                  activation_fn=activation_fn,
                  dropout=dropout,
                  batch_norm=batch_norm,
                  layer_norm=layer_norm,
                  use_resnet=use_resnet,
                  use_densenet=use_densenet,
                  is_training=is_training,
                  concat_project=concat_project)

        with tf.variable_scope('logits') as logits_scope:
            logits = fc(y, units=units, name=logits_scope)
            add_hidden_layer_summary(logits, logits_scope.name)

        return logits


def _kpnn(feature_vectors,
          project_feature_vectors,
          use_project,
          units,
          hidden_units,
          activation_fn,
          dropout,
          batch_norm,
          layer_norm,
          use_resnet,
          use_densenet,
          is_training,
          concat_project):
    """kpnn
      feature_vectors: List of shape [B, ?] tensors, size N
      project_feature_vectors: None or list of shape [B, D] tensors, size N

    Return:
      2D Tensor of shape [B, ?]
    """

    with tf.variable_scope("kpnn"):
        if use_project:
            x = project_feature_vectors
        else:
            x = feature_vectors

        y = pairwise_kernel_dot(x)  # [B, N*N]

        inputs = []
        inputs.append(y)
        if use_project and concat_project:
            inputs.extend(project_feature_vectors)
        else:
            inputs.extend(feature_vectors)

        y = tf.concat(inputs, axis=1)   # [B, ?]
        y = add_hidden_layers(y,
                              hidden_units=hidden_units,
                              activation_fn=activation_fn,
                              dropout=dropout,
                              is_training=is_training,
                              batch_norm=batch_norm,
                              layer_norm=layer_norm,
                              use_resnet=use_resnet,
                              use_densenet=use_densenet,
                              scope='hidden_layers')  # [B, ?]
        return y
