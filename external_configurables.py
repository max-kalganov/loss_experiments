"""Imports tf libs for using in config.gin"""

import gin.tf.external_configurables
from gin import config
import tensorflow as tf


config.external_configurable(tf.keras.losses.BinaryCrossentropy, module='tf.keras.losses')
config.external_configurable(tf.keras.losses.MeanSquaredError, module='tf.keras.losses')

config.external_configurable(tf.keras.metrics.BinaryAccuracy, module='tf.keras.metrics')
