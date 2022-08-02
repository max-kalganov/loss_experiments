"""Implements simple model with one layer and two nodes in it"""
import tensorflow as tf
import gin


@gin.configurable()
def get_model(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.BinaryCrossEntropy(),
              metric=tf.keras.metrics.BinaryAccuracy()):
    input = tf.keras.layers.Input(2, dtype=tf.float32)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(input)
    model = tf.keras.models.Model(inputs=[input], outputs=[x])
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[metric])
    return model
