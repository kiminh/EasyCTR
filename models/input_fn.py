#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from estimator_1_14_custom.export import export


WEIGHT_COL = '_weight_column'


def _parse_line(line, opts, scheme):
    assembler_ops = tf.load_op_library(opts.assembler_ops_path)
    feature, label, weight = assembler_ops.assembler(
        line, conf_path=opts.conf_path)
    feature_dict = _parse_singel_feature_dict(scheme, feature)
    feature_dict[WEIGHT_COL] = weight
    return (feature_dict, label)


def _parse_batch_feature_dict(scheme, features):
    feature_dict = {}
    idx = 0
    for item in scheme['feature_columns']:
        name = item['name']
        if name in feature_dict:
            raise ValueError("feature name '{}' duplicated".format(name))
        tt = item['type']
        width = item['width']
        f = features[:, idx:idx + width]
        if tt == 'int' or tt == 'string' or tt == 'string_list':
            f = tf.to_int32(f)
            feature_dict[name] = f
        elif tt == 'weighted_string_list':
            assert width % 2 == 0  # 必须是偶数
            w = int(width / 2)
            keys = f[:, :w]
            keys = tf.to_int32(keys)
            weights = tf.to_float(f[:, w:])
            feature_dict[name] = keys
            feature_dict[name + '.weight'] = weights
        elif tt == 'float' or tt == 'float_list':
            feature_dict[name] = f
        else:
            raise ValueError("Unknown feature type '{}'".format(tt))

        idx += width
    tf.logging.info('feature_dict = {}'.format(feature_dict))
    return feature_dict


def _parse_singel_feature_dict(scheme, features):
    feature_dict = {}
    idx = 0
    for item in scheme['feature_columns']:
        name = item['name']
        tt = item['type']
        width = item['width']
        f = features[idx:idx + width]
        if tt == 'int' or tt == 'string' or tt == 'string_list':
            f = tf.to_int32(f)
            feature_dict[name] = f
        elif tt == 'weighted_string_list':
            assert width % 2 == 0  # 必须是偶数
            w = int(width / 2)
            keys = f[:w]
            keys = tf.to_int32(keys)
            weights = tf.to_float(f[w:])
            feature_dict[name] = keys
            feature_dict[name + '.weight'] = weights
        else:
            feature_dict[name] = f

        idx += width
    tf.logging.info('feature_dict = {}'.format(feature_dict))
    return feature_dict


def _flat_map_example(opts, scheme, x):
    features = x[0]
    labels = x[1]
    weights = x[2]
    feature_dict = _parse_batch_feature_dict(scheme, features)
    feature_dict[WEIGHT_COL] = weights
    dataset = tf.data.Dataset.from_tensor_slices((feature_dict, labels))
    return dataset


def input_fn(opts, filenames, is_eval, scheme, epoch=1):
    batch_size = opts.eval_batch_size if is_eval else opts.batch_size

    def build_input_fn():
        if opts.read_buffer_size_mb is None:
            buffer_size = None
        else:
            buffer_size = 1024*1024*opts.read_buffer_size_mb
            buffer_size = max(1, buffer_size)

        if opts.use_spark_fuel:
            import sparkfuel as sf
            num_workers = sf.get_tf_num_workers()
            job_type, task_id = sf.get_tf_identity()
            ds = tf.data.Dataset.from_tensor_slices(filenames)
            task_id = task_id if job_type == 'chief' else task_id + 1
            ds = ds.shard(num_workers, task_id)
            ds = ds.flat_map(
                lambda filename: tf.data.TextLineDataset(
                    filename,
                    buffer_size=buffer_size,
                    compression_type=opts.compression_type))
        else:
            ds = tf.data.TextLineDataset(
                filenames,
                buffer_size=buffer_size,
                compression_type=opts.compression_type,
                num_parallel_reads=opts.num_parallel_reads)

        ds = ds.map(lambda line: _parse_line(line, opts, scheme),
                    num_parallel_calls=opts.map_num_parallel_calls)
        if opts.shuffle_batch and not is_eval:
            ds = ds.shuffle(buffer_size=opts.shuffle_size)
        if not is_eval:
            ds = ds.repeat(epoch)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(opts.prefetch_size)
        return ds

    return build_input_fn


def _parse_example(serialized, opts, scheme):
    width = 0
    for item in scheme['feature_columns']:
        width += item['width']
    example = tf.parse_single_example(
        serialized,
        features={
            'inputs': tf.FixedLenFeature([width], tf.float64),
            'label': tf.FixedLenFeature([1], tf.float64),
            'weight': tf.FixedLenFeature([1], tf.float64)
        }
    )
    feature_dict = _parse_singel_feature_dict(scheme, example['inputs'])
    feature_dict[WEIGHT_COL] = example['weight']

    return (feature_dict, example['label'])


def tfrecord_input_fn(opts, filenames, is_eval, scheme, epoch=1):
    batch_size = opts.eval_batch_size if is_eval else opts.batch_size

    def build_input_fn():
        ds = tf.data.TFRecordDataset(filenames)
        if not is_eval:
            ds = ds.apply(tf.data.experimental.shuffle_and_repeat(
                opts.shuffle_size, epoch))
        ds = ds.apply(
            tf.data.experimental.map_and_batch(
                lambda x: _parse_example(x, opts, scheme),
                batch_size,
                num_parallel_calls=opts.map_num_parallel_calls))
        ds = ds.prefetch(opts.prefetch_size)
        return ds

    return build_input_fn


def build_serving_input_fn(opts, scheme):
    assembler_ops = tf.load_op_library(opts.assembler_ops_path)
    serialized = assembler_ops.assembler_serialize(conf_path=opts.conf_path)
    with tf.Session() as sess:
        serialized = sess.run(serialized)

    def fixed_string_feature(shape):
        return tf.FixedLenFeature(shape=shape, dtype=tf.string)

    def var_string_feature():
        return tf.VarLenFeature(dtype=tf.string)

    def fixed_int64_feature(shape):
        return tf.FixedLenFeature(shape=shape, dtype=tf.int64)

    def serving_input_receiver_fn():
        feature_spec = {
            'user_feature': fixed_string_feature([1]),
            'ctx_features': var_string_feature(),
            'item_features': var_string_feature(),
        }
        serialized_tf_example = tf.placeholder(
            dtype=tf.string, shape=[None], name='input_example_tensor')
        receiver_tensors = {'examples': serialized_tf_example}
        features = tf.parse_example(serialized_tf_example, feature_spec)
        features['item_features'] = tf.sparse.to_dense(
            features['item_features'], default_value='')
        features['ctx_features'] = tf.sparse.to_dense(
            features['ctx_features'], default_value='')

        output = assembler_ops.assembler_serving(
            user_feature=features['user_feature'],
            ctx_features=features['ctx_features'],
            item_feature=features['item_features'],
            serialized=serialized)
        feature_dict = _parse_batch_feature_dict(scheme, output)
        return export.ServingInputReceiver(
            feature_dict, receiver_tensors)

    return serving_input_receiver_fn
