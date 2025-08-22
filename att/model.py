import tensorflow as tf


def att(a):
    a_ = tf.transpose(a)
    aa_ = tf.matmul(a, a_)
    score = tf.nn.softmax(aa_)
    value = tf.matmul(score, a)
    return value


def transformAttention(X, K, d, bn, bn_decay, is_training):
    """
    transform attention mechanism
    X:      [batch_size, P, N, D]
    STE_P:  [batch_size, P, N, D]
    STE_Q:  [batch_size, Q, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, Q, N, D]
    """
    D = K * d

    query = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    key = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)

    query = tf.concat(tf.split(query, D, axis=-1), axis=0)
    key = tf.concat(tf.split(key, D, axis=-1), axis=0)
    value = tf.concat(tf.split(value, D, axis=-1), axis=0)

    query = tf.transpose(query, perm=(0, 2, 1, 3))
    key = tf.transpose(key, perm=(0, 2, 3, 1))
    value = tf.transpose(value, perm=(0, 2, 1, 3))

    attention = tf.matmul(query, key)
    attention /= (D ** 0.5)
    attention = tf.nn.softmax(attention, axis=-1)

    # X = tf.matmul(attention, value)
    # X = tf.transpose(X, perm=(0, 2, 1, 3))
    # X = tf.concat(tf.split(X, K, axis=0), axis=-1)
    # X = FC(
    #     X, units=[D, D], activations=[tf.nn.relu, None],
    #     bn=bn, bn_decay=bn_decay, is_training=is_training)
    return attention


def FC(x, units, activations, bn, bn_decay, is_training, use_bias=True, drop=None):
    if isinstance(units, int):
        units = [units]
        activations = [activations]
    elif isinstance(units, tuple):
        units = list(units)
        activations = list(activations)
    assert type(units) == list
    for num_unit, activation in zip(units, activations):
        if drop is not None:
            x = dropout(x, drop=drop, is_training=is_training)
        x = conv2d(
            x, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn=bn, bn_decay=bn_decay, is_training=is_training)
    return x


def dropout(x, drop, is_training):
    x = tf.cond(
        is_training,
        lambda: tf.nn.dropout(x, rate=drop),
        lambda: x)
    return x


def conv2d(x, output_dims, kernel_size, stride=[1, 1],
           padding='SAME', use_bias=True, activation=tf.nn.relu,
           bn=False, bn_decay=None, is_training=None):
    input_dims = x.get_shape()[-1].value
    kernel_shape = kernel_size + [input_dims, output_dims]
    kernel = tf.Variable(
        tf.glorot_uniform_initializer()(shape=kernel_shape),
        dtype=tf.float32, trainable=True, name='kernel')
    x = tf.nn.conv2d(x, kernel, [1] + stride + [1], padding=padding)
    if use_bias:
        bias = tf.Variable(
            tf.zeros_initializer()(shape=[output_dims]),
            dtype=tf.float32, trainable=True, name='bias')
        x = tf.nn.bias_add(x, bias)
    if activation is not None:
        # if bn:
        #     x = batch_norm(x, is_training = is_training, bn_decay = bn_decay)
        x = activation(x)
    return x
