import numpy as np
from cnn_from_scratch import data_handling
from cnn_from_scratch import plotter
from cnn_from_scratch import cnn
from cnn_from_scratch import convolver
from cnn_from_scratch import paths

f=4
nf=2
nh=10

data = data_handling.get_MX_data(f, d=3, val_size=50, total_samples=500, small_data=True)
convolver = convolver.Convolver()
plotter = plotter.Plotter()
cnn = cnn.CNN(convolver, plotter)
cnn.set_learning_parameters()
cnn.init_network(nh, nf, f)

cnn.training_cyclical(data['MX_tr'], data['Y_tr'], data['y_tr'], data['MX_val'], data['y_val'], lam=0)

plotter.plot("Test plot cyclical", paths.PLOTS / "testfile")