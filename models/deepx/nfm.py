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
from common import get_activation_fn
from common import get_feature_vectors
from common import project
from deepx.fm import _fm


"""
Neural Factorization Machines for Sparse Predictive Analytics
"""


def _check_nfm_args(args):
    check_arg(args, 'nfm_use_shared_embedding', bool)
    check_arg(args, 'nfm_use_project', bool)
    check_arg(args, 'nfm_project_size', int)
    check_arg(args, 'nfm_hidden_units', list)
    check_arg(args, 'nfm_activation_fn', str)
    check_arg(args, 'nfm_dropout', float)
    check_arg(args, 'nfm_batch_norm', bool)
    check_arg(args, 'nfm_layer_norm', bool)
    check_arg(args, 'nfm_use_resnet', bool)
    check_arg(args, 'nfm_use_densenet', bool)

    check_arg(args, 'leaky_relu_alpha', float)
    check_arg(args, 'swish_beta', float)


def get_nfm_logits(
        features,
        feature_columns,
        shared_feature_vectors,
        units,
        is_training,
        extra_options):

    with tf.variable_scope('nfm'):
        _check_nfm_args(extra_options)
        use_shared_embedding = extra_options['nfm_use_shared_embedding']
        use_project = extra_options['nfm_use_project']
        project_size = extra_options['nfm_project_size']
        hidden_units = extra_options['nfm_hidden_units']
        activation_fn = extra_options['nfm_activation_fn']
        dropout = extra_options['nfm_dropout']
        batch_norm = extra_options['nfm_batch_norm']
        layer_norm = extra_options['nfm_layer_norm']
        use_resnet = extra_options['nfm_use_resnet']
        use_densenet = extra_options['nfm_use_densenet']

        leaky_relu_alpha = extra_options['leaky_relu_alpha']
        swish_beta = extra_options['swish_beta']

        activation_fn = get_activation_fn(activation_fn=activation_fn,
                                          leaky_relu_alpha=leaky_relu_alpha,
                                          swish_beta=swish_beta)

        if not use_shared_embedding:
            feature_vectors = get_feature_vectors(features, feature_columns)
        else:
            feature_vectors = shared_feature_vectors

        if use_project:
            feature_vectors = project(feature_vectors, project_size)

        # Neural FM
        y = _fm(feature_vectors, reduce_sum=False)
        y = add_hidden_layers(y,
                              hidden_units=hidden_units,
                              activation_fn=activation_fn,
                              dropout=dropout,
                              is_training=is_training,
                              batch_norm=batch_norm,
                              layer_norm=layer_norm,
                              use_resnet=use_resnet,
                              use_densenet=use_densenet,
                              scope='hidden_layers')
        with tf.variable_scope('logits') as logits_scope:
            logits = fc(y, units, name=logits_scope)
            add_hidden_layer_summary(logits, logits_scope.name)

        return logits
