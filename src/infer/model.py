# src/inference/model_cbam.py
from __future__ import annotations
import tensorflow as tf
import tflearn

tf.compat.v1.disable_eager_execution()

def residual_block_9x9(x, nb_blocks: int, out_ch: int, downsample: bool=False,
                       downsample_strides: int=2, act="relu", scope=None):
    """
    等價於你原始 residual_block (kernel=9)
    """
    with tf.compat.v1.variable_scope(scope or "resblk_9", reuse=tf.compat.v1.AUTO_REUSE):
        h = x
        in_ch = x.get_shape().as_list()[-1]
        for _ in range(nb_blocks):
            identity = h
            stride = 1 if not downsample else downsample_strides

            h = tflearn.batch_normalization(h)
            h = tflearn.activation(h, act)
            h = tflearn.conv_2d(h, out_ch, 9, stride, "same", "linear")

            h = tflearn.batch_normalization(h)
            h = tflearn.activation(h, act)
            h = tflearn.conv_2d(h, out_ch, 9, 1, "same", "linear")

            if stride > 1:
                identity = tflearn.avg_pool_2d(identity, 1, stride)

            if in_ch != out_ch:
                ch = (out_ch - in_ch) // 2
                identity = tf.pad(identity, [[0,0],[0,0],[0,0],[ch,ch]])
                in_ch = out_ch

            h = h + identity
        return h

def residual_block_3x3(x, nb_blocks: int, out_ch: int, downsample: bool=False,
                       downsample_strides: int=2, act="relu", scope=None):
    """
    等價於你原始 residual_block1 (kernel=3)
    """
    with tf.compat.v1.variable_scope(scope or "resblk_3", reuse=tf.compat.v1.AUTO_REUSE):
        h = x
        in_ch = x.get_shape().as_list()[-1]
        for _ in range(nb_blocks):
            identity = h
            stride = 1 if not downsample else downsample_strides

            h = tflearn.batch_normalization(h)
            h = tflearn.activation(h, act)
            h = tflearn.conv_2d(h, out_ch, 3, stride, "same", "linear")

            h = tflearn.batch_normalization(h)
            h = tflearn.activation(h, act)
            h = tflearn.conv_2d(h, out_ch, 3, 1, "same", "linear")

            if stride > 1:
                identity = tflearn.avg_pool_2d(identity, 1, stride)

            if in_ch != out_ch:
                ch = (out_ch - in_ch) // 2
                identity = tf.pad(identity, [[0,0],[0,0],[0,0],[ch,ch]])
                in_ch = out_ch

            h = h + identity
        return h

def cbam_module(inputs, reduction_ratio=0.5, scope="cbam"):
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        bs, h, w, c = inputs.get_shape().as_list()

        # Channel attention
        maxpool = tf.reduce_max(tf.reduce_max(inputs, axis=1, keepdims=True), axis=2, keepdims=True)
        avgpool = tf.reduce_mean(tf.reduce_mean(inputs, axis=1, keepdims=True), axis=2, keepdims=True)

        flat_max = tf.reshape(maxpool, [-1, c])
        flat_avg = tf.reshape(avgpool, [-1, c])

        hidden = int(c * reduction_ratio)
        with tf.compat.v1.variable_scope("mlp", reuse=tf.compat.v1.AUTO_REUSE):
            w1 = tf.compat.v1.get_variable("w1", shape=[c, hidden], initializer=tf.compat.v1.initializers.glorot_uniform())
            b1 = tf.compat.v1.get_variable("b1", shape=[hidden], initializer=tf.zeros_initializer())
            w2 = tf.compat.v1.get_variable("w2", shape=[hidden, c], initializer=tf.compat.v1.initializers.glorot_uniform())
            b2 = tf.compat.v1.get_variable("b2", shape=[c], initializer=tf.zeros_initializer())

            m1 = tf.nn.relu(tf.matmul(flat_max, w1) + b1)
            m1 = tf.matmul(m1, w2) + b2

            a1 = tf.nn.relu(tf.matmul(flat_avg, w1) + b1)
            a1 = tf.matmul(a1, w2) + b2

        ca = tf.nn.sigmoid(tf.reshape(m1 + a1, [-1, 1, 1, c]))
        x = inputs * ca

        # Spatial attention
        max_sp = tf.reduce_max(x, axis=3, keepdims=True)
        avg_sp = tf.reduce_mean(x, axis=3, keepdims=True)
        sp = tf.concat([max_sp, avg_sp], axis=3)
        sa = tflearn.conv_2d(sp, 1, 7, 1, "same", "linear")
        sa = tf.nn.sigmoid(sa)

        return x * sa

def instance_norm(x):
    return tf.contrib.layers.instance_norm(x, center=True, scale=True, epsilon=1e-6, data_format="NHWC")

def deconv(x, out_ch, k, stride, name):
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        return tflearn.conv_2d_transpose(x, out_ch, k, stride, "same", "linear")

def build_cbam_autoencoder(inputs, is_training=False):
    """
    重現你的前向圖：
      res9(63) -> res3(63,down) -> res9(127,down)
      + CBAM -> 三層 deconv (9,3,9)  → sigmoid
    inputs: 4-D (N, H, W, 3), 已縮放到 0~1
    """
    x = inputs
    l1 = residual_block_9x9(x, 1, 63, downsample=False, scope="res9_1")
    l2 = residual_block_3x3(l1, 1, 63, downsample=True, scope="res3_ds")
    l3 = residual_block_9x9(l2, 1, 127, downsample=True, scope="res9_ds")

    cbam = cbam_module(l3, reduction_ratio=0.5, scope="cbam")
    feat = l3 + cbam

    h1 = tf.nn.relu(instance_norm(deconv(feat, 128, 9, 1, "deconv1")))   # 64x64x128
    h2 = tf.nn.relu(instance_norm(deconv(h1, 64, 3, 2, "deconv2")))      # 128x128x64
    h3 = deconv(h2, 3, 9, 2, "deconv3")                                  # 256x256x3
    out = tf.nn.sigmoid(h3, name="output")
    return out
