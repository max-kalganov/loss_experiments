"""Implements simple model with one layer and two nodes in it"""
from typing import Optional, List

import tensorflow as tf
import gin


@gin.configurable()
def get_model(optimizer, loss, metric, kernel_init: Optional[List[float]] = None):
    x_input = tf.keras.layers.Input(2, dtype=tf.float32)
    kernel_init = kernel_init if kernel_init is None else tf.constant_initializer(kernel_init)
    x = tf.keras.layers.Dense(1,
                              activation='sigmoid',
                              kernel_initializer=kernel_init,
                              bias_initializer=tf.keras.initializers.Zeros())(x_input)
    model = tf.keras.models.Model(inputs=[x_input], outputs=[x])
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[metric])
    return model
