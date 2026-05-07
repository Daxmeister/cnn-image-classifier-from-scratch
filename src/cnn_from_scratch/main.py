import numpy as np
from cnn_from_scratch import data_handling
from cnn_from_scratch import plotter
from cnn_from_scratch import cnn
from cnn_from_scratch import convolver
from cnn_from_scratch import paths
import time


def train_and_eval_net(f=4, nf=10, nh=50):
    data = data_handling.get_MX_data(f, d=3, val_size=1000, small_data=False)

    conv = convolver.Convolver()
    plot_device = plotter.Plotter()
    cnn_net = cnn.CNN(conv, plotter)
    cnn_net.set_learning_parameters(n_batch=100, eta_min = 1e-5, eta_max = 1e-1, n_s=800, n_cycles=3)
    cnn_net.init_network(nh, nf, f)

    start = time.time()
    cnn_net.training_cyclical(data['MX_tr'], data['Y_tr'], data['y_tr'], data['MX_val'], data['y_val'], lam=0.003)
    end = time.time()

    plot_device.plot("Test plot cyclical", paths.PLOTS / "testfile")
    print(cnn_net.evaluate_accuracy(data['MX_test'], data['y_te']))

def train_and_measure_time(f=4, nf=10, nh=50):
    data = data_handling.get_MX_data(f, d=3, val_size=1000, small_data=False)

    conv = convolver.Convolver()

    cnn_net = cnn.CNN(conv)
    cnn_net.set_learning_parameters(n_batch=100, eta_min = 1e-5, eta_max = 1e-1, n_s=800, n_cycles=3)
    cnn_net.init_network(nh, nf, f)

    start_time = time.time()
    cnn_net.training_cyclical(data['MX_tr'], data['Y_tr'], data['y_tr'], data['MX_val'], data['y_val'], lam=0.003)
    end_time = time.time()

    print(f"Settings: f={f}, nf={nf}, nh={nh}")
    print(cnn_net.evaluate_accuracy(data['MX_test'], data['y_te']))
    print(f"Time elapsed: {end_time-start_time:.2f}")

train_and_measure_time()