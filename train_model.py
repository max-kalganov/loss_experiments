"""Loss function visualization and experiments using config"""
from data_generator import get_dataset
import gin
import gin.tf
from model import get_model


@gin.configurable()
def run_exp(epochs: int, loss):
    x, y = get_dataset()
    x_train, x_test, y_train, y_test = x[:(len(x) // 2), :], x[(len(x) // 2):, :], y[:(len(y) // 2)], y[(len(y) // 2):]

    model = get_model(loss=loss)
    print(model.summary())

    model.fit(x_train, y_train, epochs=epochs, validation=[x_test, y_test])


if __name__ == '__main__':
    gin.parse_config_file("config.gin")
    run_exp()
