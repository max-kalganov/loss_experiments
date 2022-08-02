"""Loss function visualization and experiments using config"""
from data_generator import get_dataset
import gin
from model import get_model

if __name__ == '__main__':
    gin.config.parse_config()
    x, y = get_dataset()
    x_train, x_test, y_train, y_test = x[:(len(x)//2), :], x[(len(x)//2):, :], y[:(len(y)//2)], y[(len(y)//2):]

    model = get_model()
