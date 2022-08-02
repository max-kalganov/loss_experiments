"""Loss function visualization and experiments using config"""
from data_generator import get_dataset
import gin
import gin.tf
from model import get_model
import external_configurables
import tensorflow as tf


class PrintLayer(tf.keras.callbacks.Callback):
    def __round(self, val):
        return (val.numpy().squeeze() * 100).round() / 100

    def on_epoch_end(self, *args, **kwargs):
        target_layer = self.model.layers[1]
        print(f"\nkernel={self.__round(target_layer.kernel)}, "
              f"bias={self.__round(target_layer.bias)}")



@gin.configurable()
def run_exp(epochs: int, loss):
    x, y = get_dataset()
    x_train, x_test, y_train, y_test = x[:(len(x) // 2), :], x[(len(x) // 2):, :], y[:(len(y) // 2)], y[(len(y) // 2):]

    model = get_model(loss=loss)
    print(model.summary())

    model.fit(x_train, y_train, epochs=epochs, validation_data=[x_test, y_test], callbacks=[PrintLayer()])
    print()


if __name__ == '__main__':
    gin.parse_config_file("config.gin")
    run_exp()
