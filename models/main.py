#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import json
import tensorflow as tf

from estimator_1_14_custom.canned.dnn_linear_combined import DNNLinearCombinedRegressor
from estimator_1_14_custom.canned.dnn import DNNRegressor
from estimator_1_14_custom.canned.linear import LinearRegressor
from estimator_1_14_custom.run_config import RunConfig

from bi_tempered_loss import bi_tempered_binary_logistic_loss
from focal_loss import binary_focal_loss

import transform
import input_fn
import hook
from yellowfin import YFOptimizer
from dssm import dssm
from deepx import deepx
from common import get_file_content


parser = argparse.ArgumentParser()


def DEFINE_string(flag, default, description):
    parser.add_argument(
        "--" + flag,
        type=str,
        default=default,
        help=description
    )


def DEFINE_integer(flag, default, description):
    parser.add_argument(
        "--" + flag,
        type=int,
        default=default,
        help=description
    )


def DEFINE_float(flag, default, description):
    parser.add_argument(
        "--" + flag,
        type=float,
        default=default,
        help=description
    )


def DEFINE_bool(flag, default, description):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument(
        "--" + flag,
        type=str2bool,
        default=default,
        help=description
    )


def DEFINE_list(flag, default, description):
    def str2list(v):
        return v.split(',')

    parser.add_argument(
        "--" + flag,
        type=str2list,
        default=default,
        help=description
    )


DEFINE_string('model_dir', '', '')
DEFINE_string('export_model_dir', '', '')
DEFINE_string('dict_dir', '', '')
DEFINE_bool('use_tfrecord', False, '')
DEFINE_bool('do_profile', False, '')
DEFINE_integer('profile_every_steps', 1000, '')
DEFINE_string('conf_path', '', 'conf path')
DEFINE_string('assembler_ops_path', '', 'assembler_ops_path')

DEFINE_bool('do_train', True, '')
DEFINE_bool('do_eval', True, '')
DEFINE_bool('do_export', True, '')
DEFINE_string('train_data_path', '', 'train data path')
DEFINE_string('eval_data_path', '', 'eval data path')

DEFINE_string('model_type', 'wide_deep', '')
DEFINE_string('dssm_mode', 'dot', 'dot, concat, cosine')
DEFINE_list('model_slots', 'dnn', '')
DEFINE_string('extend_feature_mode', '', 'fgcnn')

# dssm
DEFINE_list('dssm_hidden_units', '512,256,128', '')
DEFINE_string('dssm_activation_fn', 'relu', '')
DEFINE_float('dssm_dropout', 0.0, '')
DEFINE_bool('dssm_batch_norm', False, '')
DEFINE_bool('dssm_layer_norm', False, '')
DEFINE_bool('dssm_use_resnet', False, '')
DEFINE_bool('dssm_use_densenet', False, '')

# deepx models' options
DEFINE_bool('use_seperated_logits', False, '')
DEFINE_bool('use_weighted_logits', False, '')

# fm
DEFINE_bool('fm_use_shared_embedding', True, '')
DEFINE_bool('fm_use_project', False, '')
DEFINE_integer('fm_project_size', 32, '')

# fwfm
DEFINE_bool('fwfm_use_shared_embedding', True, '')
DEFINE_bool('fwfm_use_project', False, '')
DEFINE_integer('fwfm_project_size', 32, '')

# afm
DEFINE_bool('afm_use_shared_embedding', True, '')
DEFINE_bool('afm_use_project', False, '')
DEFINE_integer('afm_project_size', 32, '')
DEFINE_integer('afm_hidden_unit', 32, '')

# iafm
DEFINE_bool('iafm_use_shared_embedding', True, '')
DEFINE_bool('iafm_use_project', False, '')
DEFINE_integer('iafm_project_size', 32, '')
DEFINE_integer('iafm_hidden_unit', 32, '')
DEFINE_integer('iafm_field_dim', 8, '')

# ifm
DEFINE_bool('ifm_use_shared_embedding', True, '')
DEFINE_bool('ifm_use_project', False, '')
DEFINE_integer('ifm_project_size', 32, '')
DEFINE_integer('ifm_hidden_unit', 32, '')
DEFINE_integer('ifm_field_dim', 8, '')

# kfm
DEFINE_bool('kfm_use_shared_embedding', True, '')
DEFINE_bool('kfm_use_project', False, '')
DEFINE_integer('kfm_project_size', 32, '')

# wkfm
DEFINE_bool('wkfm_use_shared_embedding', True, '')
DEFINE_bool('wkfm_use_project', False, '')
DEFINE_integer('wkfm_project_size', 32, '')

# nifm
DEFINE_bool('nifm_use_shared_embedding', True, '')
DEFINE_bool('nifm_use_project', False, '')
DEFINE_integer('nifm_project_size', 32, '')
DEFINE_list('nifm_hidden_units', '32,16', '')
DEFINE_string('nifm_activation_fn', 'relu', '')
DEFINE_float('nifm_dropout', 0.0, '')
DEFINE_bool('nifm_batch_norm', False, '')
DEFINE_bool('nifm_layer_norm', False, '')
DEFINE_bool('nifm_use_resnet', False, '')
DEFINE_bool('nifm_use_densenet', False, '')

# cin
DEFINE_bool('cin_use_shared_embedding', True, '')
DEFINE_bool('cin_use_project', False, '')
DEFINE_integer('cin_project_size', 32, '')
DEFINE_list('cin_hidden_feature_maps', '128,128', '')
DEFINE_bool('cin_split_half', True, '')

# cross
DEFINE_bool('cross_use_shared_embedding', True, '')
DEFINE_bool('cross_use_project', False, '')
DEFINE_integer('cross_project_size', 32, '')
DEFINE_integer('cross_num_layers', 4, '')

# autoint
DEFINE_bool('autoint_use_shared_embedding', True, '')
DEFINE_bool('autoint_use_project', False, '')
DEFINE_integer('autoint_project_size', 32, '')
DEFINE_integer('autoint_size_per_head', 16, '')
DEFINE_integer('autoint_num_heads', 6, '')
DEFINE_integer('autoint_num_blocks', 2, '')
DEFINE_float('autoint_dropout', 0.0, '')
DEFINE_bool('autoint_has_residual', True, '')

# dnn
DEFINE_bool('dnn_use_shared_embedding', True, '')
DEFINE_bool('dnn_use_project', False, '')
DEFINE_integer('dnn_project_size', 32, '')
DEFINE_list('dnn_hidden_units', '512,256,128', '')
DEFINE_string('dnn_activation_fn', 'relu', '')
DEFINE_float('dnn_dropout', 0.0, '')
DEFINE_bool('dnn_batch_norm', False, '')
DEFINE_bool('dnn_layer_norm', False, '')
DEFINE_bool('dnn_use_resnet', False, '')
DEFINE_bool('dnn_use_densenet', False, '')

# nfm
DEFINE_bool('nfm_use_shared_embedding', True, '')
DEFINE_bool('nfm_use_project', False, '')
DEFINE_integer('nfm_project_size', 32, '')
DEFINE_list('nfm_hidden_units', '512,256,128', '')
DEFINE_string('nfm_activation_fn', 'relu', '')
DEFINE_float('nfm_dropout', 0.0, '')
DEFINE_bool('nfm_batch_norm', False, '')
DEFINE_bool('nfm_layer_norm', False, '')
DEFINE_bool('nfm_use_resnet', False, '')
DEFINE_bool('nfm_use_densenet', False, '')

# nkfm
DEFINE_bool('nkfm_use_shared_embedding', True, '')
DEFINE_bool('nkfm_use_project', False, '')
DEFINE_integer('nkfm_project_size', 32, '')
DEFINE_list('nkfm_hidden_units', '512,256,128', '')
DEFINE_string('nkfm_activation_fn', 'relu', '')
DEFINE_float('nkfm_dropout', 0.0, '')
DEFINE_bool('nkfm_batch_norm', False, '')
DEFINE_bool('nkfm_layer_norm', False, '')
DEFINE_bool('nkfm_use_resnet', False, '')
DEFINE_bool('nkfm_use_densenet', False, '')

# ccpm
DEFINE_bool('ccpm_use_shared_embedding', True, '')
DEFINE_bool('ccpm_use_project', False, '')
DEFINE_integer('ccpm_project_size', 32, '')
DEFINE_list('ccpm_hidden_units', '512,256,128', '')
DEFINE_string('ccpm_activation_fn', 'relu', '')
DEFINE_float('ccpm_dropout', 0.0, '')
DEFINE_bool('ccpm_batch_norm', False, '')
DEFINE_bool('ccpm_layer_norm', False, '')
DEFINE_bool('ccpm_use_resnet', False, '')
DEFINE_bool('ccpm_use_densenet', False, '')
DEFINE_list('ccpm_kernel_sizes', '3,3,3', '')
DEFINE_list('ccpm_filter_nums', '4,3,2', '')

# ipnn
DEFINE_bool('ipnn_use_shared_embedding', True, '')
DEFINE_bool('ipnn_use_project', False, '')
DEFINE_integer('ipnn_project_size', 32, '')
DEFINE_list('ipnn_hidden_units', '512,256,128', '')
DEFINE_string('ipnn_activation_fn', 'relu', '')
DEFINE_float('ipnn_dropout', 0.0, '')
DEFINE_bool('ipnn_batch_norm', False, '')
DEFINE_bool('ipnn_layer_norm', False, '')
DEFINE_bool('ipnn_use_resnet', False, '')
DEFINE_bool('ipnn_use_densenet', False, '')
DEFINE_bool('ipnn_unordered_inner_product', False, '')
DEFINE_bool('ipnn_concat_project', False, '')

# kpnn
DEFINE_bool('kpnn_use_shared_embedding', True, '')
DEFINE_bool('kpnn_use_project', False, '')
DEFINE_integer('kpnn_project_size', 32, '')
DEFINE_list('kpnn_hidden_units', '512,256,128', '')
DEFINE_string('kpnn_activation_fn', 'relu', '')
DEFINE_float('kpnn_dropout', 0.0, '')
DEFINE_bool('kpnn_batch_norm', False, '')
DEFINE_bool('kpnn_layer_norm', False, '')
DEFINE_bool('kpnn_use_resnet', False, '')
DEFINE_bool('kpnn_use_densenet', False, '')
DEFINE_bool('kpnn_concat_project', False, '')

# pin
DEFINE_bool('pin_use_shared_embedding', True, '')
DEFINE_bool('pin_use_project', False, '')
DEFINE_integer('pin_project_size', 32, '')
DEFINE_list('pin_hidden_units', '512,256,128', '')
DEFINE_string('pin_activation_fn', 'relu', '')
DEFINE_float('pin_dropout', 0.0, '')
DEFINE_bool('pin_batch_norm', False, '')
DEFINE_bool('pin_layer_norm', False, '')
DEFINE_bool('pin_use_resnet', False, '')
DEFINE_bool('pin_use_densenet', False, '')
DEFINE_bool('pin_use_concat', False, '')
DEFINE_bool('pin_concat_project', False, '')
DEFINE_list('pin_subnet_hidden_units', '64,32', '')

# fibinet
DEFINE_bool('fibinet_use_shared_embedding', True, '')
DEFINE_bool('fibinet_use_project', False, '')
DEFINE_integer('fibinet_project_size', 32, '')
DEFINE_list('fibinet_hidden_units', '512,256,128', '')
DEFINE_string('fibinet_activation_fn', 'relu', '')
DEFINE_float('fibinet_dropout', 0.0, '')
DEFINE_bool('fibinet_batch_norm', False, '')
DEFINE_bool('fibinet_layer_norm', False, '')
DEFINE_bool('fibinet_use_resnet', False, '')
DEFINE_bool('fibinet_use_densenet', False, '')
DEFINE_bool('fibinet_use_se', True, '')
DEFINE_bool('fibinet_use_deep', True, '')
DEFINE_string(
    'fibinet_interaction_type', 'bilinear',
    '"inner", "hadamard", "bilinear"')
DEFINE_string(
    'fibinet_se_interaction_type', 'bilinear',
    '"inner", "hadamard", "bilinear"')
DEFINE_bool('fibinet_se_use_shared_embedding', False, '')

# fgcnn
DEFINE_bool('fgcnn_use_shared_embedding', False, '')
DEFINE_bool('fgcnn_use_project', False, '')
DEFINE_integer('fgcnn_project_dim', 32, '')
DEFINE_list('fgcnn_filter_nums', '6,8', '')
DEFINE_list('fgcnn_kernel_sizes', '7,7', '')
DEFINE_list('fgcnn_pooling_sizes', '2,2', '')
DEFINE_list('fgcnn_new_map_sizes', '3,3', '')

# activation functions
DEFINE_float('leaky_relu_alpha', 0.2, '')
DEFINE_float('swish_beta', 1.0, '')

# train flags
DEFINE_string('loss_reduction', 'sum', '"mean", "sum"')
DEFINE_string('loss_fn', 'ce', '"ce", "bi_tempered", "focal"')
DEFINE_float('bi_tempered_loss_t1', 1.0, '')
DEFINE_float('bi_tempered_loss_t2', 1.0, '')
DEFINE_float('bi_tempered_loss_label_smoothing', 0.0, '')
DEFINE_integer('bi_tempered_loss_num_iters', 5, '')
DEFINE_float('focal_loss_gamma', 2.0, '')
DEFINE_string('linear_sparse_combiner', 'sum', 'sum or mean')
DEFINE_integer('batch_size', 512, 'batch size')
DEFINE_integer('eval_batch_size', 1024, 'eval batch size')
DEFINE_integer('max_train_steps', -1, '')
DEFINE_integer('epoch', 1, '')
DEFINE_integer(
    'total_steps', 1,
    'total train steps inferenced from train data')
DEFINE_integer('num_gpus', 0, 'num gpus')

# dataset flags
DEFINE_integer('prefetch_size', 4096, '')
DEFINE_integer('shuffle_size', 2000, '')
DEFINE_bool('shuffle_batch', False, '')
DEFINE_integer('map_num_parallel_calls', 10, '')
DEFINE_integer(
    'num_parallel_reads', 1,
    ' the number of files to read in parallel.')
DEFINE_integer('read_buffer_size_mb', 1024, '')

# log flags
DEFINE_integer('save_summary_steps', 1000, '')
DEFINE_integer('save_checkpoints_steps', 100000, '')
DEFINE_integer('keep_checkpoint_max', 3, '')
DEFINE_integer('log_step_count_steps', 1000, '')
DEFINE_string('serving_warmup_file', '', '')

# optimizer
DEFINE_string(
    'deep_optimizer', 'adagrad',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam", '
    '"ftrl", "momentum", "sgd", "rmsprop", "proximal_adagrad", '
    '"yellowfin", "adamw".')
DEFINE_float('deep_learning_rate', 0.1, 'learning rate')

DEFINE_float(
    'deep_adadelta_rho', 0.95,
    'The decay rate for adadelta.')

DEFINE_float(
    'deep_adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

DEFINE_float(
    'deep_adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

DEFINE_float(
    'deep_adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

DEFINE_float(
    'deep_opt_epsilon', 1e-6,
    'Epsilon term for the optimizer.')

DEFINE_float('deep_ftrl_learning_rate_power', -0.5,
             'deep_The learning rate power.')

DEFINE_float(
    'deep_ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

DEFINE_float(
    'deep_ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

DEFINE_float(
    'deep_ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

DEFINE_float(
    'deep_momentum', 0.9,
    'The momentum for the MomentumOptimizer, RMSPropOptimizer and YFOptimizer.')

DEFINE_float('deep_rmsprop_momentum', 0.9, 'Momentum.')
DEFINE_float('deep_rmsprop_decay', 0.9, 'Decay term for RMSProp.')
DEFINE_float(
    'deep_proximal_adagrad_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

DEFINE_float(
    'deep_proximal_adagrad_l1', 0.0,
    'The ProximalAdagrad l1 regularization strength.')

DEFINE_float(
    'deep_proximal_adagrad_l2', 0.0,
    'The ProximalAdagrad l2 regularization strength.')

DEFINE_float('deep_adamw_weight_decay_rate', 0.01, '')
DEFINE_float('deep_adamw_beta1', 0.9, '')
DEFINE_float('deep_adamw_beta2', 0.999, '')

DEFINE_string(
    'deep_learning_rate_decay_type',
    'fixed',
    'Specifies how the learning rate is decayed. One of "fixed", '
    '"exponential", "polynomial", or "warmup"')

DEFINE_float(
    'deep_end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

DEFINE_integer(
    'deep_decay_steps', 1000,
    'The decay steps used by a polynomial decay learning rate.')

DEFINE_float(
    'deep_learning_rate_decay_factor', 0.99,
    'The decay rate used by a exponential decay learning rate.')
DEFINE_float('deep_cosine_decay_alpha', 0.0, '')
DEFINE_float('deep_polynomial_decay_power', 1.0, '')
DEFINE_float('deep_warmup_rate', 0.3, '')

DEFINE_string(
    'wide_optimizer', 'ftrl',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd", "rmsprop", "proximal_adagrad", "yellowfin".')
DEFINE_float('wide_learning_rate', 0.005, 'learning rate')

DEFINE_float(
    'wide_adadelta_rho', 0.95,
    'The decay rate for adadelta.')

DEFINE_float(
    'wide_adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

DEFINE_float(
    'wide_adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

DEFINE_float(
    'wide_adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

DEFINE_float(
    'wide_opt_epsilon', 1e-6,
    'Epsilon term for the optimizer.')

DEFINE_float(
    'wide_ftrl_learning_rate_power', -0.5,
    'The learning rate power.')

DEFINE_float(
    'wide_ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

DEFINE_float(
    'wide_ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

DEFINE_float(
    'wide_ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

DEFINE_float(
    'wide_momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

DEFINE_float('wide_rmsprop_momentum', 0.9, 'Momentum.')
DEFINE_float('wide_rmsprop_decay', 0.9, 'Decay term for RMSProp.')

DEFINE_float(
    'wide_proximal_adagrad_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

DEFINE_float(
    'wide_proximal_adagrad_l1', 0.0,
    'The ProximalAdagrad l1 regularization strength.')

DEFINE_float(
    'wide_proximal_adagrad_l2', 0.0,
    'The ProximalAdagrad l2 regularization strength.')

DEFINE_string(
    'wide_learning_rate_decay_type',
    'fixed',
    'Specifies how the learning rate is decayed. One of "fixed", '
    '"exponential", or "polynomial"')

DEFINE_float(
    'wide_end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

DEFINE_integer(
    'wide_decay_steps', 1000,
    'The decay steps used by a polynomial decay learning rate.')

DEFINE_float(
    'wide_learning_rate_decay_factor', 0.99,
    'The decay rate used by a exponential decay learning rate.')
DEFINE_float('wide_polynomial_decay_power', 1.0, '')

DEFINE_string('compression_type', '', '"", "GZIP"')

DEFINE_string(
    'result_output_file', 'result.log',
    'output evaluate result to a file')

DEFINE_float('auc_thr', 0.6, 'auc threshold.')

DEFINE_bool('use_spark_fuel', False, '')
DEFINE_integer('num_ps', 1, 'Number of ps nodes.')
DEFINE_integer('start_delay_secs', 120, '')
DEFINE_integer('throttle_secs', 600, '')
DEFINE_integer('eval_steps_every_checkpoint', 100, '')

# TODO(zhezhaoxu) Local mode not support it, now only used in spark fuel mode
DEFINE_bool('remove_model_dir', True, '')


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'


# scheme 是由 assembler op 生成的一个 json 对象
def parse_scheme(conf_path, ops_path, use_spark_fuel):
    assembler_ops = tf.load_op_library(ops_path)
    with tf.Session() as sess:
        output = assembler_ops.assembler_scheme(conf_path=conf_path)
        scheme = sess.run(output)
    scheme = json.loads(scheme)

    return scheme


def configure_deep_optimizer(opts):
    """Configures the deep optimizer used for training."""

    learning_rate = configure_deep_learning_rate(opts)
    tf.summary.scalar('deep_learning_rate', learning_rate)
    if opts.deep_optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=opts.deep_adadelta_rho,
            epsilon=opts.deep_opt_epsilon)
    elif opts.deep_optimizer == 'adagrad':
        init_value = opts.deep_adagrad_initial_accumulator_value
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=init_value)
    elif opts.deep_optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=opts.deep_adam_beta1,
            beta2=opts.deep_adam_beta2,
            epsilon=opts.deep_opt_epsilon)
    elif opts.deep_optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=opts.deep_ftrl_learning_rate_power,
            initial_accumulator_value=opts.deep_ftrl_initial_accumulator_value,
            l1_regularization_strength=opts.deep_ftrl_l1,
            l2_regularization_strength=opts.deep_ftrl_l2)
    elif opts.deep_optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=opts.deep_momentum,
            name='Momentum')
    elif opts.deep_optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=opts.deep_rmsprop_decay,
            momentum=opts.deep_rmsprop_momentum,
            epsilon=opts.deep_opt_epsilon)
    elif opts.deep_optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif opts.deep_optimizer == 'proximal_adagrad':
        optimizer = tf.train.ProximalAdagradOptimizer(
            learning_rate,
            initial_accumulator_value=opts.deep_proximal_adagrad_initial_accumulator_value,
            l1_regularization_strength=opts.deep_proximal_adagrad_l1,
            l2_regularization_strength=opts.deep_proximal_adagrad_l2)
    elif opts.deep_optimizer == 'yellowfin':
        optimizer = YFOptimizer(
            learning_rate,
            momentum=opts.deep_momentum)
    elif opts.deep_optimizer == 'adamw':
        if opts.num_gpus > 1:
            from optimization_multi_gpu import AdamWeightDecayOptimizer
        else:
            from optimization import AdamWeightDecayOptimizer

        optimizer = AdamWeightDecayOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=opts.deep_adamw_weight_decay_rate,
            beta_1=opts.deep_adamw_beta1,
            beta_2=opts.deep_adamw_beta2,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias",
                                       "BatchNorm", "batch_norm"])
    else:
        raise ValueError('Optimizer [%s] was not recognized'
                         % opts.deep_optimizer)
    return optimizer


def configure_wide_optimizer(opts):
    """Configures the wide optimizer used for training."""

    learning_rate = configure_wide_learning_rate(opts)
    tf.summary.scalar('wide_learning_rate', learning_rate)
    if opts.wide_optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=opts.wide_adadelta_rho,
            epsilon=opts.wide_opt_epsilon)
    elif opts.wide_optimizer == 'adagrad':
        init_value = opts.deep_adagrad_initial_accumulator_value
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=init_value)
    elif opts.wide_optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=opts.wide_adam_beta1,
            beta2=opts.wide_adam_beta2,
            epsilon=opts.wide_opt_epsilon)
    elif opts.wide_optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=opts.wide_ftrl_learning_rate_power,
            initial_accumulator_value=opts.wide_ftrl_initial_accumulator_value,
            l1_regularization_strength=opts.wide_ftrl_l1,
            l2_regularization_strength=opts.wide_ftrl_l2)
    elif opts.wide_optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=opts.wide_momentum,
            name='Momentum')
    elif opts.wide_optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=opts.wide_rmsprop_decay,
            momentum=opts.wide_rmsprop_momentum,
            epsilon=opts.wide_opt_epsilon)
    elif opts.wide_optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif opts.wide_optimizer == 'proximal_adagrad':
        optimizer = tf.train.ProximalAdagradOptimizer(
            learning_rate,
            initial_accumulator_value=opts.wide_proximal_adagrad_initial_accumulator_value,
            l1_regularization_strength=opts.wide_proximal_adagrad_l1,
            l2_regularization_strength=opts.wide_proximal_adagrad_l2)
    elif opts.wide_optimizer == 'yellowfin':
        optimizer = YFOptimizer(
            learning_rate,
            momentum=opts.wide_momentum)
    else:
        raise ValueError('Optimizer [%s] was not recognized'
                         % opts.wide_optimizer)
    return optimizer


def warmup_learning_rate(init_lr, warmup_rate, global_step, total_steps):
    assert total_steps > 0, 'total_steps must be larger than 0'
    assert warmup_rate > 0, 'warmup_rate must be larger than 0'

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        total_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)
    global_steps_int = tf.cast(global_step, tf.int32)
    num_warmup_steps = int(warmup_rate * total_steps)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    return learning_rate


def configure_deep_learning_rate(opts):
    global_step = tf.train.get_global_step()
    if opts.deep_learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(
            opts.deep_learning_rate,
            global_step,
            opts.deep_decay_steps,
            opts.deep_learning_rate_decay_factor,
            staircase=True,
            name='deep_exponential_decay_learning_rate')
    elif opts.deep_learning_rate_decay_type == 'fixed':
        return tf.constant(opts.deep_learning_rate,
                           name='deep_fixed_learning_rate')
    elif opts.deep_learning_rate_decay_type == 'polynomial':
        decay_steps = opts.deep_decay_steps
        if opts.deep_decay_steps <= 0:
            decay_steps = opts.total_steps
        return tf.train.polynomial_decay(
            opts.deep_learning_rate,
            global_step,
            decay_steps,
            opts.deep_end_learning_rate,
            power=opts.deep_polynomial_decay_power,
            cycle=False,
            name='deep_polynomial_decay_learning_rate')
    elif opts.deep_learning_rate_decay_type == 'warmup':
        learning_rate = warmup_learning_rate(
            init_lr=opts.deep_learning_rate,
            warmup_rate=opts.deep_warmup_rate,
            global_step=global_step,
            total_steps=opts.total_steps)
        return learning_rate
    elif opts.deep_learning_rate_decay_type == 'cosine':
        decay_steps = opts.deep_decay_steps
        if opts.deep_decay_steps <= 0:
            decay_steps = opts.total_steps
        return tf.train.cosine_decay(
            learning_rate=opts.deep_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            alpha=opts.deep_cosine_decay_alpha)
    else:
        raise ValueError('deep_learning_rate_decay_type [{}] was not recognized'
                         .format(opts.deep_learning_rate_decay_type))


def configure_wide_learning_rate(opts):
    global_step = tf.train.get_global_step()
    if opts.wide_learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(
            opts.wide_learning_rate,
            global_step,
            opts.wide_decay_steps,
            opts.wide_learning_rate_decay_factor,
            staircase=True,
            name='wide_exponential_decay_learning_rate')
    elif opts.wide_learning_rate_decay_type == 'fixed':
        return tf.constant(opts.wide_learning_rate,
                           name='wide_fixed_learning_rate')
    elif opts.wide_learning_rate_decay_type == 'polynomial':
        decay_steps = opts.wide_decay_steps
        if opts.wide_decay_steps <= 0:
            decay_steps = opts.total_steps

        return tf.train.polynomial_decay(
            opts.wide_learning_rate,
            global_step,
            decay_steps,
            opts.wide_end_learning_rate,
            power=opts.wide_polynomial_decay_power,
            cycle=False,
            name='wide_polynomial_decay_learning_rate')
    else:
        raise ValueError('wide_learning_rate_decay_type [{}] was not recognized'
                         .format(opts.wide_learning_rate_decay_type))


def configure_loss_fn(opts):
    if opts.loss_fn == 'ce':
        loss_fn = None
    elif opts.loss_fn == 'bi_tempered':
        """Robust Bi-Tempered Logistic Loss Based on Bregman Divergences"""

        def loss_wrapper(labels, logits):
            return bi_tempered_binary_logistic_loss(
                activations=logits,  # 根据论文, activations 就是网络 logits
                labels=labels,
                t1=opts.bi_tempered_loss_t1,
                t2=opts.bi_tempered_loss_t2,
                label_smoothing=opts.bi_tempered_loss_label_smoothing,
                num_iters=opts.bi_tempered_loss_num_iters)
        loss_fn = loss_wrapper
    elif opts.loss_fn == 'focal':
        """Focal Loss"""

        def loss_wrapper(labels, logits):
            return binary_focal_loss(
                labels=labels,
                logits=logits,
                gamma=opts.focal_loss_gamma)
        loss_fn = loss_wrapper
    else:
        raise ValueError("Unknown loss_fn '{}'".format(opts.loss_fn))
    return loss_fn


def get_config(opts):
    config_keys = {}
    config_keys['model_dir'] = opts.model_dir
    config_keys['tf_random_seed'] = None
    config_keys['save_summary_steps'] = opts.save_summary_steps
    config_keys['save_checkpoints_steps'] = opts.save_checkpoints_steps
    config_keys['session_config'] = None
    config_keys['keep_checkpoint_max'] = opts.keep_checkpoint_max
    config_keys['keep_checkpoint_every_n_hours'] = 10000
    config_keys['log_step_count_steps'] = opts.log_step_count_steps
    if opts.num_gpus > 1:
        distribution = tf.contrib.distribute.MirroredStrategy(
            num_gpus=opts.num_gpus)
        config_keys['train_distribute'] = distribution
    run_config = RunConfig(**config_keys)

    return run_config


def get_optimizer(opts):
    if opts.wide_optimizer == 'default':
        wide_optimizer = 'Ftrl'
    else:
        def wide_optimizer():
            return configure_wide_optimizer(opts)

    if opts.deep_optimizer == 'default':
        deep_optimizer = 'Adagrad'
    else:
        def deep_optimizer():
            return configure_deep_optimizer(opts)

    return wide_optimizer, deep_optimizer


def build_estimator(opts, scheme, conf):
    trans = transform.Transform(conf, scheme)

    wide_columns = trans.get_wide_columns()
    deep_columns = trans.get_deep_columns()
    # TODO(zhezhaoxu) add attention
    # attention_columns = trans.get_attention_columns()
    dssm1_columns = trans.get_dssm1_columns()
    dssm2_columns = trans.get_dssm2_columns()

    run_config = get_config(opts)
    wide_optimizer, deep_optimizer = get_optimizer(opts)
    loss_fn = configure_loss_fn(opts)
    if opts.loss_reduction == 'mean':
        loss_reduction = tf.losses.Reduction.MEAN
    elif opts.loss_reduction == 'sum':
        loss_reduction = tf.losses.Reduction.SUM
    else:
        raise ValueError("Unknown loss_reduction '{}'".format(loss_reduction))

    extra_options = {}
    extra_options['leaky_relu_alpha'] = opts.leaky_relu_alpha
    extra_options['swish_beta'] = opts.swish_beta

    extra_options['fm_use_shared_embedding'] = opts.fm_use_shared_embedding
    extra_options['fm_use_project'] = opts.fm_use_project
    extra_options['fm_project_size'] = opts.fm_project_size

    extra_options['fwfm_use_shared_embedding'] = opts.fwfm_use_shared_embedding
    extra_options['fwfm_use_project'] = opts.fwfm_use_project
    extra_options['fwfm_project_size'] = opts.fwfm_project_size

    extra_options['afm_use_shared_embedding'] = opts.afm_use_shared_embedding
    extra_options['afm_use_project'] = opts.afm_use_project
    extra_options['afm_project_size'] = opts.afm_project_size
    extra_options['afm_hidden_unit'] = opts.afm_hidden_unit

    extra_options['iafm_use_shared_embedding'] = opts.iafm_use_shared_embedding
    extra_options['iafm_use_project'] = opts.iafm_use_project
    extra_options['iafm_project_size'] = opts.iafm_project_size
    extra_options['iafm_hidden_unit'] = opts.iafm_hidden_unit
    extra_options['iafm_field_dim'] = opts.iafm_field_dim

    extra_options['ifm_use_shared_embedding'] = opts.ifm_use_shared_embedding
    extra_options['ifm_use_project'] = opts.ifm_use_project
    extra_options['ifm_project_size'] = opts.ifm_project_size
    extra_options['ifm_hidden_unit'] = opts.ifm_hidden_unit
    extra_options['ifm_field_dim'] = opts.ifm_field_dim

    extra_options['kfm_use_shared_embedding'] = opts.kfm_use_shared_embedding
    extra_options['kfm_use_project'] = opts.kfm_use_project
    extra_options['kfm_project_size'] = opts.kfm_project_size

    extra_options['wkfm_use_shared_embedding'] = opts.wkfm_use_shared_embedding
    extra_options['wkfm_use_project'] = opts.wkfm_use_project
    extra_options['wkfm_project_size'] = opts.wkfm_project_size

    extra_options['nifm_use_shared_embedding'] = opts.nifm_use_shared_embedding
    extra_options['nifm_use_project'] = opts.nifm_use_project
    extra_options['nifm_project_size'] = opts.nifm_project_size
    extra_options['nifm_hidden_units'] = [int(x) for x in opts.nifm_hidden_units]
    extra_options['nifm_activation_fn'] = opts.nifm_activation_fn
    extra_options['nifm_dropout'] = opts.nifm_dropout
    extra_options['nifm_batch_norm'] = opts.nifm_batch_norm
    extra_options['nifm_layer_norm'] = opts.nifm_layer_norm
    extra_options['nifm_use_resnet'] = opts.nifm_use_resnet
    extra_options['nifm_use_densenet'] = opts.nifm_use_densenet

    extra_options['cin_use_shared_embedding'] = opts.cin_use_shared_embedding
    extra_options['cin_use_project'] = opts.cin_use_project
    extra_options['cin_project_size'] = opts.cin_project_size
    extra_options['cin_hidden_feature_maps'] = [int(x) for x in opts.cin_hidden_feature_maps]
    extra_options['cin_split_half'] = opts.cin_split_half

    extra_options['cross_use_shared_embedding'] = opts.cross_use_shared_embedding
    extra_options['cross_use_project'] = opts.cross_use_project
    extra_options['cross_project_size'] = opts.cross_project_size
    extra_options['cross_num_layers'] = opts.cross_num_layers

    extra_options['autoint_use_shared_embedding'] = opts.autoint_use_shared_embedding
    extra_options['autoint_use_project'] = opts.autoint_use_project
    extra_options['autoint_project_size'] = opts.autoint_project_size
    extra_options['autoint_size_per_head'] = opts.autoint_size_per_head
    extra_options['autoint_num_heads'] = opts.autoint_num_heads
    extra_options['autoint_num_blocks'] = opts.autoint_num_blocks
    extra_options['autoint_dropout'] = opts.autoint_dropout
    extra_options['autoint_has_residual'] = opts.autoint_has_residual

    extra_options['dnn_use_shared_embedding'] = opts.dnn_use_shared_embedding
    extra_options['dnn_use_project'] = opts.dnn_use_project
    extra_options['dnn_project_size'] = opts.dnn_project_size
    extra_options['dnn_hidden_units'] = [int(x) for x in opts.dnn_hidden_units]
    extra_options['dnn_activation_fn'] = opts.dnn_activation_fn
    extra_options['dnn_dropout'] = opts.dnn_dropout
    extra_options['dnn_batch_norm'] = opts.dnn_batch_norm
    extra_options['dnn_layer_norm'] = opts.dnn_layer_norm
    extra_options['dnn_use_resnet'] = opts.dnn_use_resnet
    extra_options['dnn_use_densenet'] = opts.dnn_use_densenet

    extra_options['nfm_use_shared_embedding'] = opts.nfm_use_shared_embedding
    extra_options['nfm_use_project'] = opts.nfm_use_project
    extra_options['nfm_project_size'] = opts.nfm_project_size
    extra_options['nfm_hidden_units'] = [int(x) for x in opts.nfm_hidden_units]
    extra_options['nfm_activation_fn'] = opts.nfm_activation_fn
    extra_options['nfm_dropout'] = opts.nfm_dropout
    extra_options['nfm_batch_norm'] = opts.nfm_batch_norm
    extra_options['nfm_layer_norm'] = opts.nfm_layer_norm
    extra_options['nfm_use_resnet'] = opts.nfm_use_resnet
    extra_options['nfm_use_densenet'] = opts.nfm_use_densenet

    extra_options['nkfm_use_shared_embedding'] = opts.nkfm_use_shared_embedding
    extra_options['nkfm_use_project'] = opts.nkfm_use_project
    extra_options['nkfm_project_size'] = opts.nkfm_project_size
    extra_options['nkfm_hidden_units'] = [int(x) for x in opts.nkfm_hidden_units]
    extra_options['nkfm_activation_fn'] = opts.nkfm_activation_fn
    extra_options['nkfm_dropout'] = opts.nkfm_dropout
    extra_options['nkfm_batch_norm'] = opts.nkfm_batch_norm
    extra_options['nkfm_layer_norm'] = opts.nkfm_layer_norm
    extra_options['nkfm_use_resnet'] = opts.nkfm_use_resnet
    extra_options['nkfm_use_densenet'] = opts.nkfm_use_densenet

    extra_options['ccpm_use_shared_embedding'] = opts.ccpm_use_shared_embedding
    extra_options['ccpm_use_project'] = opts.ccpm_use_project
    extra_options['ccpm_project_size'] = opts.ccpm_project_size
    extra_options['ccpm_hidden_units'] = [int(x) for x in opts.ccpm_hidden_units]
    extra_options['ccpm_activation_fn'] = opts.ccpm_activation_fn
    extra_options['ccpm_dropout'] = opts.ccpm_dropout
    extra_options['ccpm_batch_norm'] = opts.ccpm_batch_norm
    extra_options['ccpm_layer_norm'] = opts.ccpm_layer_norm
    extra_options['ccpm_use_resnet'] = opts.ccpm_use_resnet
    extra_options['ccpm_use_densenet'] = opts.ccpm_use_densenet
    extra_options['ccpm_kernel_sizes'] = [int(x) for x in opts.ccpm_kernel_sizes]
    extra_options['ccpm_filter_nums'] = [int(x) for x in opts.ccpm_filter_nums]

    extra_options['ipnn_use_shared_embedding'] = opts.ipnn_use_shared_embedding
    extra_options['ipnn_use_project'] = opts.ipnn_use_project
    extra_options['ipnn_project_size'] = opts.ipnn_project_size
    extra_options['ipnn_hidden_units'] = [int(x) for x in opts.ipnn_hidden_units]
    extra_options['ipnn_activation_fn'] = opts.ipnn_activation_fn
    extra_options['ipnn_dropout'] = opts.ipnn_dropout
    extra_options['ipnn_batch_norm'] = opts.ipnn_batch_norm
    extra_options['ipnn_layer_norm'] = opts.ipnn_layer_norm
    extra_options['ipnn_use_resnet'] = opts.ipnn_use_resnet
    extra_options['ipnn_use_densenet'] = opts.ipnn_use_densenet
    extra_options['ipnn_unordered_inner_product'] = opts.ipnn_unordered_inner_product
    extra_options['ipnn_concat_project'] = opts.ipnn_concat_project

    extra_options['kpnn_use_shared_embedding'] = opts.kpnn_use_shared_embedding
    extra_options['kpnn_use_project'] = opts.kpnn_use_project
    extra_options['kpnn_project_size'] = opts.kpnn_project_size
    extra_options['kpnn_hidden_units'] = [int(x) for x in opts.kpnn_hidden_units]
    extra_options['kpnn_activation_fn'] = opts.kpnn_activation_fn
    extra_options['kpnn_dropout'] = opts.kpnn_dropout
    extra_options['kpnn_batch_norm'] = opts.kpnn_batch_norm
    extra_options['kpnn_layer_norm'] = opts.kpnn_layer_norm
    extra_options['kpnn_use_resnet'] = opts.kpnn_use_resnet
    extra_options['kpnn_use_densenet'] = opts.kpnn_use_densenet
    extra_options['kpnn_concat_project'] = opts.kpnn_concat_project

    extra_options['pin_use_shared_embedding'] = opts.pin_use_shared_embedding
    extra_options['pin_use_project'] = opts.pin_use_project
    extra_options['pin_project_size'] = opts.pin_project_size
    extra_options['pin_hidden_units'] = [int(x) for x in opts.pin_hidden_units]
    extra_options['pin_activation_fn'] = opts.pin_activation_fn
    extra_options['pin_dropout'] = opts.pin_dropout
    extra_options['pin_batch_norm'] = opts.pin_batch_norm
    extra_options['pin_layer_norm'] = opts.pin_layer_norm
    extra_options['pin_use_resnet'] = opts.pin_use_resnet
    extra_options['pin_use_densenet'] = opts.pin_use_densenet
    extra_options['pin_use_concat'] = opts.pin_use_concat
    extra_options['pin_concat_project'] = opts.pin_concat_project
    extra_options['pin_subnet_hidden_units'] = [int(x) for x in opts.pin_subnet_hidden_units]

    extra_options['fibinet_use_shared_embedding'] = opts.fibinet_use_shared_embedding
    extra_options['fibinet_use_project'] = opts.fibinet_use_project
    extra_options['fibinet_project_size'] = opts.fibinet_project_size
    extra_options['fibinet_hidden_units'] = [int(x) for x in opts.fibinet_hidden_units]
    extra_options['fibinet_activation_fn'] = opts.fibinet_activation_fn
    extra_options['fibinet_dropout'] = opts.fibinet_dropout
    extra_options['fibinet_batch_norm'] = opts.fibinet_batch_norm
    extra_options['fibinet_layer_norm'] = opts.fibinet_layer_norm
    extra_options['fibinet_use_resnet'] = opts.fibinet_use_resnet
    extra_options['fibinet_use_densenet'] = opts.fibinet_use_densenet
    extra_options['fibinet_use_se'] = opts.fibinet_use_se
    extra_options['fibinet_use_deep'] = opts.fibinet_use_deep
    extra_options['fibinet_interaction_type'] = opts.fibinet_interaction_type
    extra_options['fibinet_se_interaction_type'] = opts.fibinet_se_interaction_type
    extra_options['fibinet_se_use_shared_embedding'] = opts.fibinet_se_use_shared_embedding

    extra_options['fgcnn_use_shared_embedding'] = opts.fgcnn_use_shared_embedding
    extra_options['fgcnn_use_project'] = opts.fgcnn_use_project
    extra_options['fgcnn_project_dim'] = opts.fgcnn_project_dim
    extra_options['fgcnn_filter_nums'] = [int(x) for x in opts.fgcnn_filter_nums]
    extra_options['fgcnn_kernel_sizes'] = [int(x) for x in opts.fgcnn_kernel_sizes]
    extra_options['fgcnn_pooling_sizes'] = [int(x) for x in opts.fgcnn_pooling_sizes]
    extra_options['fgcnn_new_map_sizes'] = [int(x) for x in opts.fgcnn_new_map_sizes]

    model_type = opts.model_type
    model_slots = opts.model_slots

    # just for convenience
    model_slots_map = {
        'wide': ['linear'],
        'lr': ['linear'],
        'linear': ['linear'],
        'dnn': ['dnn'],
        'deep': ['dnn'],
        'wide_deep': ['dnn', 'linear'],
        'wdl': ['dnn', 'linear'],
        'fm': ['fm', 'linear'],
        'fwfm': ['fwfm', 'linear'],
        'afm': ['afm', 'linear'],
        'iafm': ['iafm', 'linear'],
        'ifm': ['ifm', 'linear'],
        'kfm': ['kfm', 'linear'],
        'wkfm': ['wkfm', 'linear'],
        'nifm': ['nifm', 'linear'],
        'cross': ['cross', 'linear'],
        'cin': ['cin', 'linear'],
        'autoint': ['autoint', 'linear'],
        'autoint+': ['autoint', 'dnn', 'linear'],
        'nfm': ['nfm', 'linear'],
        'nkfm': ['nkfm', 'linear'],
        'ccpm': ['ccpm', 'linear'],
        'ipnn': ['ipnn', 'linear'],
        'kpnn': ['kpnn', 'linear'],
        'pin': ['pin', 'linear'],
        'fibinet': ['fibinet', 'linear'],
        'deepfm': ['dnn', 'fm', 'linear'],
        'deepfwfm': ['dnn', 'fwfm', 'linear'],
        'deepafm': ['dnn', 'afm', 'linear'],
        'deepiafm': ['dnn', 'iafm', 'linear'],
        'deepifm': ['dnn', 'ifm', 'linear'],
        'deepkfm': ['dnn', 'kfm', 'linear'],
        'deepwkfm': ['dnn', 'wkfm', 'linear'],
        'deepnifm': ['dnn', 'nifm', 'linear'],
        'xdeepfm': ['dnn', 'cin', 'linear'],
        'dcn': ['dnn', 'cross', 'linear'],
        'deepcross': ['dnn', 'cross', 'linear'],
    }

    if model_type in model_slots_map:
        model_slots = model_slots_map[model_type]
        model_type = 'deepx'

    if model_type == 'dssm':
        estimator = dssm.DSSMEstimator(
            hidden_units=opts.dssm_hidden_units,
            activation_fn=opts.dssm_activation_fn,
            leaky_relu_alpha=opts.leaky_relu_alpha,
            swish_beta=opts.swish_beta,
            dropout=opts.dssm_dropout,
            batch_norm=opts.dssm_batch_norm,
            layer_norm=opts.dssm_layer_norm,
            use_resnet=opts.dssm_use_resnet,
            use_densenet=opts.dssm_use_densenet,
            dssm1_columns=dssm1_columns,
            dssm2_columns=dssm2_columns,
            dssm_mode=opts.dssm_mode,
            model_dir=opts.model_dir,
            n_classes=2,
            weight_column=input_fn.WEIGHT_COL,
            optimizer=deep_optimizer,
            config=run_config,
            loss_fn=loss_fn,
            loss_reduction=loss_reduction)
    elif model_type == 'deepx':
        estimator = deepx.DeepXClassifier(
            model_dir=opts.model_dir,
            linear_feature_columns=wide_columns,
            linear_optimizer=wide_optimizer,
            deep_feature_columns=deep_columns,
            deep_optimizer=deep_optimizer,
            model_slots=model_slots,
            extend_feature_mode=opts.extend_feature_mode,
            n_classes=2,
            weight_column=input_fn.WEIGHT_COL,
            config=run_config,
            linear_sparse_combiner=opts.linear_sparse_combiner,
            loss_fn=loss_fn,
            loss_reduction=loss_reduction,
            use_seperated_logits=opts.use_seperated_logits,
            use_weighted_logits=opts.use_weighted_logits,
            extra_options=extra_options,
        )
    elif model_type == 'wide_regress':
        estimator = LinearRegressor(
            feature_columns=wide_columns,
            model_dir=opts.model_dir,
            weight_column=input_fn.WEIGHT_COL,
            optimizer=wide_optimizer,
            config=run_config,
            sparse_combiner=opts.linear_sparse_combiner,
            loss_fn=loss_fn,
            loss_reduction=loss_reduction)
    elif model_type == 'deep_regress':
        estimator = DNNRegressor(
            hidden_units=extra_options['dnn_hidden_units'],
            feature_columns=deep_columns,
            model_dir=opts.model_dir,
            weight_column=input_fn.WEIGHT_COL,
            activation_fn=extra_options['dnn_activation_fn'],
            dropout=extra_options['dnn_dropout'],
            optimizer=deep_optimizer,
            config=run_config,
            loss_fn=loss_fn,
            loss_reduction=loss_reduction)
    elif model_type == 'wide_deep_regress':
        estimator = DNNLinearCombinedRegressor(
            model_dir=opts.model_dir,
            linear_feature_columns=wide_columns,
            linear_optimizer=wide_optimizer,
            dnn_feature_columns=deep_columns,
            dnn_optimizer=deep_optimizer,
            dnn_hidden_units=extra_options['dnn_hidden_units'],
            dnn_activation_fn=extra_options['dnn_activation_fn'],
            dnn_dropout=extra_options['dnn_dropout'],
            weight_column=input_fn.WEIGHT_COL,
            config=run_config,
            batch_norm=opts.batch_norm,
            linear_sparse_combiner=opts.linear_sparse_combiner,
            loss_fn=loss_fn,
            loss_reduction=loss_reduction)
    else:
        raise ValueError("Unknow model_type '{}'".format(opts.model_type))

    tf.logging.info("estimator: {}".format(estimator))
    return estimator


def create_profile_hooks(save_steps, model_dir):
    """Create profile hooks."""

    meta_hook = hook.MetadataHook(
        save_steps=save_steps, output_dir=model_dir)
    profile_hook = tf.train.ProfilerHook(
        save_steps=save_steps,
        output_dir=model_dir,
        show_dataflow=True,
        show_memory=True)
    return meta_hook, profile_hook


def do_train(opts, scheme, conf, train_files):
    tf.logging.info("Training model ...")
    estimator = build_estimator(opts, scheme, conf)
    max_train_steps = None if opts.max_train_steps < 0 else opts.max_train_steps
    if opts.use_tfrecord:
        train_input_fn = input_fn.tfrecord_input_fn(
            opts,
            filenames=train_files,
            is_eval=False,
            scheme=scheme,
            epoch=opts.epoch)
    else:
        train_input_fn = input_fn.input_fn(
            opts,
            filenames=train_files,
            is_eval=False,
            scheme=scheme,
            epoch=opts.epoch)

    hooks = None
    if opts.do_profile:
        hooks = create_profile_hooks(opts.profile_every_steps, opts.model_dir)

    estimator.train(
        input_fn=train_input_fn,
        max_steps=max_train_steps,
        hooks=hooks)
    tf.logging.info("Training model done")


def do_evaluate(opts, scheme, conf, eval_files):
    tf.logging.info("Evaluating model ...")
    estimator = build_estimator(opts, scheme, conf)
    if opts.use_tfrecord:
        eval_input_fn = input_fn.tfrecord_input_fn(
            opts,
            filenames=eval_files,
            is_eval=True)
    else:
        eval_input_fn = input_fn.input_fn(
            opts,
            filenames=eval_files,
            is_eval=True,
            scheme=scheme)
    result = estimator.evaluate(input_fn=eval_input_fn)

    if 'auc' in result:
        if result['auc'] < opts.auc_thr:
            raise ValueError("oops, auc less than {}".format(opts.auc_thr))

    tf.logging.info("evaluate result:")
    with open(opts.result_output_file, 'w') as f:
        for key in result:
            try:
                value = float(result[key])
            except Exception:
                value = result[key]
            if isinstance(value, float):
                value = format(value, '.7g')
            s = '{} = {}'.format(key, value)
            tf.logging.info(s)
            f.write(s + '\n')

    tf.logging.info("Evaluating model done")


def do_export(opts, scheme, conf):
    tf.logging.info("Export model ...")
    estimator = build_estimator(opts, scheme, conf)
    assets_extra = {}
    assets_extra['tf_serving_warmup_requests'] = opts.serving_warmup_file
    assets_extra['conf_file'] = opts.conf_path
    assets_extra['result_output_file'] = opts.result_output_file

    if opts.dict_dir != '':
        dict_files = [line for line in os.listdir(opts.dict_dir)
                      if line.endswith('.dict')]
        for dict_file in dict_files:
            assets_extra[dict_file] = os.path.join(opts.dict_dir, dict_file)

    estimator.export_saved_model(
        opts.export_model_dir,
        serving_input_receiver_fn=input_fn.build_serving_input_fn(opts, scheme),
        assets_extra=assets_extra)
    tf.logging.info("Export model done")


def do_spark_fuel_run(opts, scheme, conf, train_files, eval_files):
    estimator = build_estimator(opts, scheme, conf)
    train_input_fn = input_fn.input_fn(
        opts,
        filenames=train_files,
        is_eval=False,
        scheme=scheme,
        epoch=opts.epoch)

    eval_input_fn = input_fn.input_fn(
        opts,
        filenames=eval_files,
        is_eval=True,
        scheme=scheme)

    if opts.do_train:
        # TODO(zhezhaoxu) Officially, train_and_eval only support max_steps stop
        # condition
        max_train_steps = None if opts.max_train_steps < 0 else opts.max_train_steps
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                            max_steps=max_train_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                          steps=opts.eval_steps_every_checkpoint,
                                          exporters=None,
                                          start_delay_secs=opts.start_delay_secs,
                                          throttle_secs=opts.throttle_secs)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if opts.do_eval:
        import sparkfuel as sf
        job_type, _ = sf.get_tf_identity()
        tf.logging.info("job_type = {}".format(job_type))
        if job_type == 'chief':
            tf.logging.info("Evaluating test dataset ...")
            eval_input_fn = input_fn.input_fn(
                opts,
                filenames=eval_files,
                is_eval=True,
                scheme=scheme)
            estimator.evaluate(input_fn=eval_input_fn)
            tf.logging.info("Evaluating test dataset done")
        else:
            tf.logging.info("Do not do evaluation in non-chief node")

    if opts.do_export:
        import sparkfuel as sf
        job_type, _ = sf.get_tf_identity()
        tf.logging.info("job_type = {}".format(job_type))
        if job_type == 'chief':
            tf.logging.info("Exporting model ...")
            serving_fn = input_fn.build_serving_input_fn(opts, scheme)
            estimator.export_saved_model(
                opts.export_model_dir,
                serving_input_receiver_fn=serving_fn)
            tf.logging.info("Export model done")
        else:
            tf.logging.info("Do not do export in non-chief node")


def run(opts, conf, train_files, eval_files):
    scheme = parse_scheme(opts.conf_path,
                          opts.assembler_ops_path,
                          opts.use_spark_fuel)
    if opts.use_spark_fuel:
        # In distributed mode, we must use train_and_eval
        do_spark_fuel_run(opts, scheme, conf, train_files, eval_files)
    else:
        if opts.do_train:
            do_train(opts, scheme, conf, train_files)
        if opts.do_eval:
            do_evaluate(opts, scheme, conf, eval_files)
        if opts.do_export:
            do_export(opts, scheme, conf)


if __name__ == '__main__':
    opts = parser.parse_args()

    if tf.__version__ != '1.14.0':
        raise ValueError("Sorry, you must use tf 1.14.0")
    # 防止日志重复打印两次
    # https://stackoverflow.com/questions/33662648/tensorflow-causes-logging-messages-to-double
    logger = tf.get_logger()
    logger.propagate = False
    tf.logging.set_verbosity(tf.logging.INFO)

    spark = None
    if opts.use_spark_fuel:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()

    # we must read hdfs file before dive into spark-fuel
    conf = get_file_content(opts.conf_path, opts.use_spark_fuel, spark)
    conf = json.loads(conf)

    train_files = get_file_content(opts.train_data_path, opts.use_spark_fuel, spark)
    train_files = train_files.split('\n')
    train_files = [x.strip() for x in train_files if x.strip() != '']
    tf.logging.info("train files = {}".format(train_files))

    eval_files = get_file_content(opts.eval_data_path, opts.use_spark_fuel, spark)
    eval_files = eval_files.split('\n')
    eval_files = [x.strip() for x in eval_files if x.strip() != '']
    tf.logging.info("eval files = {}".format(eval_files))

    if opts.use_spark_fuel:
        import sparkfuel as sf
        from common import hdfs_remove_dir
        if opts.remove_model_dir:
            ok = hdfs_remove_dir(spark, opts.model_dir)
            if ok:
                tf.logging.info("remove model dir '{}' done".format(opts.model_dir))
            else:
                tf.logging.info("remove model dir '{}' failed".format(opts.model_dir))

        with sf.TFSparkSession(spark, num_ps=opts.num_ps) as sess:
            sess.run(run, opts, conf, train_files, eval_files)
    else:
        run(opts, conf, train_files, eval_files)
