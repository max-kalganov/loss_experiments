"""Implements simple model with one layer and two nodes in it"""
import tensorflow as tf
import gin


@gin.configurable()
def get_model(optimizer, loss, metric):
    x_input = tf.keras.layers.Input(2, dtype=tf.float32)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x_input)
    model = tf.keras.models.Model(inputs=[x_input], outputs=[x])
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[metric])
    return model
