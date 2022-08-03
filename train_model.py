"""Loss function visualization and experiments using config"""
from itertools import product
from typing import Tuple

import numpy as np

from data_generator import get_dataset
import gin
import gin.tf
from model import get_model
import external_configurables
import tensorflow as tf
from plotly import graph_objects as go
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from tqdm import tqdm


class PrintLayer(tf.keras.callbacks.Callback):
    def __round(self, val):
        return (val.numpy().squeeze() * 100).round() / 100

    def on_epoch_end(self, *args, **kwargs):
        target_layer = self.model.layers[1]
        print(f"\nkernel={self.__round(target_layer.kernel)}, "
              f"bias={self.__round(target_layer.bias)}")


def train_model(x, y, epochs, loss):
    border = len(x) * 0.7
    x_train, x_test, y_train, y_test = x[:border, :], x[border:, :], y[:border], y[border:]

    model = get_model(loss=loss)
    print(model.summary())

    model.fit(x_train, y_train, epochs=epochs, validation_data=[x_test, y_test], callbacks=[PrintLayer()])
    return model


@gin.configurable()
def visualize_loss(x, y, weights_range_tuple: Tuple, loss):
    loss_values = []
    weights_range = np.arange(*weights_range_tuple)
    input_values = np.array(list(product(weights_range, weights_range)))
    for w1, w2 in tqdm(input_values):
        model = get_model(loss=loss, kernel_init=[w1, w2])
        y_pred = model.predict(x, verbose=0)
        loss_values.append(loss(y_true=y, y_pred=y_pred).numpy())

    loss_values = np.array(loss_values)
    loss_map = loss_values.reshape((len(weights_range), len(weights_range)))

    fig = go.Figure(data=[go.Surface(x=weights_range, y=weights_range, z=loss_map)])

    fig.update_layout(title='Loss function for different weights', autosize=True,
                      margin=dict(l=65, r=50, b=65, t=90))

    fig.write_html(file=f'data/loss_surface_'
                        f'{weights_range_tuple[0]}_'
                        f'{weights_range_tuple[1]}_'
                        f'{weights_range_tuple[2]}.html')
    fig.show()



@gin.configurable()
def run_exp(epochs: int, loss):
    x, y = get_dataset()

    visualize_loss(x=x, y=y, loss=loss)
    #model = train_model(x, y, epochs, loss)


if __name__ == '__main__':
    gin.parse_config_file("config.gin")
    run_exp()
