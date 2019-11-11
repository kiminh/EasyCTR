#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from common import add_hidden_layer_summary
from common import add_hidden_layers
from common import get_activation_fn

from estimator_1_14_custom import estimator
from estimator_1_14_custom.canned import head as head_lib
from estimator_1_14_custom.canned import optimizers


_LEARNING_RATE = 0.05


class DSSMEstimator(estimator.Estimator):
    """An estimator for DSSM model.
    """

    def __init__(
        self,
        hidden_units,
        activation_fn,
        leaky_relu_alpha,
        swish_beta,
        dropout,
        batch_norm,
        layer_norm,
        use_resnet,
        use_densenet,
        dssm1_columns,
        dssm2_columns,
        model_dir=None,
        dssm_mode='dot',
        n_classes=2,
        weight_column=None,
        label_vocabulary=None,
        optimizer='Adagrad',
        input_layer_partitioner=None,
        config=None,
        warm_start_from=None,
        loss_reduction=tf.losses.Reduction.SUM,
        loss_fn=None
    ):

        def _model_fn(features, labels, mode, config):
            head = head_lib._binary_logistic_or_multi_class_head(  # pylint: disable=protected-access
                n_classes, weight_column, label_vocabulary, loss_reduction, loss_fn)
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
            net_dssm1 = tf.feature_column.input_layer(features, dssm1_columns)
            net_dssm2 = tf.feature_column.input_layer(features, dssm2_columns)
            tf.logging.info("net_dssm1: {}".format(net_dssm1))
            tf.logging.info("net_dssm2: {}".format(net_dssm2))
            real_activation_fn = get_activation_fn(
                activation_fn=activation_fn,
                leaky_relu_alpha=leaky_relu_alpha,
                swish_beta=swish_beta)

            net_dssm1 = add_hidden_layers(
                inputs=net_dssm1,
                hidden_units=hidden_units,
                activation_fn=real_activation_fn,
                dropout=dropout,
                is_training=is_training,
                batch_norm=batch_norm,
                layer_norm=layer_norm,
                use_resnet=use_resnet,
                use_densenet=use_densenet,
                scope='dssm1')
            net_dssm2 = add_hidden_layers(
                inputs=net_dssm2,
                hidden_units=hidden_units,
                activation_fn=real_activation_fn,
                dropout=dropout,
                is_training=is_training,
                batch_norm=batch_norm,
                layer_norm=layer_norm,
                use_resnet=use_resnet,
                use_densenet=use_densenet,
                scope='dssm2')

            with tf.variable_scope('logits') as logits_scope:
                if dssm_mode == 'dot':
                    logits = tf.reduce_sum(net_dssm1*net_dssm2, -1, keepdims=True)
                elif dssm_mode == 'concat':
                    logits = tf.concat([net_dssm1, net_dssm2], axis=1)
                    logits = tf.layers.dense(logits, units=1, activation=None)
                elif dssm_mode == 'cosine':
                    logits = tf.reduce_sum(net_dssm1*net_dssm2, -1, keepdims=True)
                    norm1 = tf.norm(net_dssm1, axis=1, keepdims=True)
                    norm2 = tf.norm(net_dssm2, axis=1, keepdims=True)
                    logits = logits / (norm1 * norm2)
                else:
                    raise ValueError("unknown dssm mode '{}'".format(dssm_mode))
                add_hidden_layer_summary(logits, logits_scope.name)

            tf.logging.info("logits = {}".format(logits))

            return head.create_estimator_spec(
                features=features,
                mode=mode,
                labels=labels,
                optimizer=optimizers.get_optimizer_instance(
                    optimizer,
                    learning_rate=_LEARNING_RATE),
                logits=logits)

        super(DSSMEstimator, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config,
            warm_start_from=warm_start_from)
