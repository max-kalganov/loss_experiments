"""Implements simple model with one layer and two nodes in it"""
from typing import Optional, List

import tensorflow as tf
import gin
# tf.keras.backend.set_floatx('float64')
tf.config.set_soft_device_placement(True)
tf.random.set_seed(0)


@gin.configurable()
def get_model(optimizer, loss, metrics, kernel_init: Optional[List[float]] = None):
    x_input = tf.keras.layers.Input(2, dtype=tf.float64)
    kernel_init = kernel_init if kernel_init is None else tf.keras.initializers.Constant(kernel_init)
    x = tf.keras.layers.Dense(1,
                              activation='sigmoid',
                              kernel_initializer=kernel_init,
                              bias_initializer=tf.keras.initializers.Zeros())(x_input)
    model = tf.keras.models.Model(inputs=[x_input], outputs=[x])
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    return model


@gin.configurable()
def get_bigger_model(optimizer, loss, metrics, kernel_init: Optional[List[float]] = None):
    x_input = tf.keras.layers.Input(2, dtype=tf.float64)
    kernel_init = kernel_init if kernel_init is None else tf.keras.initializers.Constant(kernel_init)
    x = tf.keras.layers.Dense(2, activation='relu')(x_input)
    x = tf.keras.layers.Dense(3, activation='relu')(x)
    x = tf.keras.layers.Dense(2, activation='relu')(x)
    x = tf.keras.layers.Dense(1,
                              activation='sigmoid',
                              kernel_initializer=kernel_init,
                              bias_initializer=tf.keras.initializers.Zeros())(x)
    model = tf.keras.models.Model(inputs=[x_input], outputs=[x])
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    return model
