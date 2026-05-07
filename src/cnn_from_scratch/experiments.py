import numpy as np
import matplotlib.pyplot as plt
from cnn_from_scratch import data_handling
from cnn_from_scratch import plotter
from cnn_from_scratch import cnn
from cnn_from_scratch import convolver
from cnn_from_scratch import paths
import time




def train_and_measure_time(f=4, nf=10, nh=50):
    data = data_handling.get_MX_data(f, d=3, val_size=1000, small_data=False)

    conv = convolver.Convolver()

    cnn_net = cnn.CNN(conv)
    cnn_net.set_learning_parameters(n_batch=100, eta_min = 1e-5, eta_max = 1e-1, n_s=800, n_cycles=3)
    cnn_net.init_network(nh, nf, f)

    start_time = time.time()
    cnn_net.training_cyclical(data['MX_tr'], data['Y_tr'], data['y_tr'], data['MX_val'], data['y_val'], lam=0.003)
    end_time = time.time()

    accuracy = cnn_net.evaluate_accuracy(data['MX_test'], data['y_te'])
    train_time = end_time-start_time
    print(f"Settings: f={f}, nf={nf}, nh={nh}")
    print(accuracy)
    print(f"Time elapsed: {end_time-start_time:.2f}")
    return accuracy, train_time


def do_ex_3_1():
    accuracy_array = []
    time_array = []
    architecture_names = []
    

    architectures = [
        {"name": "Arch 1: f=2, nf=3, nh=50", "f": 2, "nf": 3, "nh": 50},
        {"name": "Arch 2: f=4, nf=10, nh=50", "f": 4, "nf": 10, "nh": 50},
        {"name": "Arch 3: f=8, nf=40, nh=50", "f": 8, "nf": 40, "nh": 50},
        {"name": "Arch 4: f=16, nf=160, nh=50", "f": 16, "nf": 160, "nh": 50},
    ]

    for arch in architectures:
        acc, train_time = train_and_measure_time(
            f=arch["f"],
            nf=arch["nf"],
            nh=arch["nh"]
        )

        accuracy_array.append(acc)
        time_array.append(train_time)
        architecture_names.append(arch["name"])
        
        
        
    
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        
        bars_acc = axs[0].bar(architecture_names, accuracy_array)
        axs[0].bar_label(bars_acc, fmt='%.3f')
        axs[0].set_xlabel('Architecture number')
        axs[0].set_ylabel('Final test accuracy')
        axs[0].set_title("Ex3 test accuracies")
        axs[0].tick_params(axis="x", rotation=20)
        #axs[0].legend()
        
        
        bars_time = axs[1].bar(architecture_names, time_array)
        axs[1].bar_label(bars_time, fmt='%.1f s')
        axs[1].set_xlabel('Architecture number')
        axs[1].set_ylabel("Training time (s)")
        axs[1].set_title("Ex3 training times")
        axs[1].tick_params(axis="x", rotation=20)
        #axs[1].legend()
        
        fig.tight_layout()
        fig.savefig(paths.PLOTS / "ii_plots.png")
        plt.close(fig)
   



def train_and_eval_net_increaseing_cycle(f=4, nf=10, nh=50, plot_filename="iii_plots.png", 
                                         n_cycles=3, n_s=800, lam=0.003,
                                         use_label_smoothing=False, epsilon=0.1):
    
    data = data_handling.get_MX_data(f, d=3, val_size=1000, small_data=False, use_label_smoothing=use_label_smoothing, epsilon=epsilon)

    conv = convolver.Convolver()
    plot_device = plotter.Plotter()
    cnn_net = cnn.CNN(conv, plot_device)
    cnn_net.set_learning_parameters(n_batch=100, eta_min = 1e-5, eta_max = 1e-1, n_s=n_s, n_cycles=n_cycles)
    cnn_net.init_network(nh, nf, f)

    cnn_net.training_cyclical_increasing_cycle_length(data['MX_tr'], data['Y_tr'], data['y_tr'], data['MX_test'], data['y_te'], lam=lam)


    plot_device.plot("Test plot cyclical", paths.PLOTS / plot_filename)
    print("Accuracy of network with increased cycles and parameters",nh, nf, f)
    print(cnn_net.evaluate_accuracy(data['MX_test'], data['y_te']))

    
def do_ex_3_2():
    architectures = [
        #{"name": "Arch2", "f": 4, "nf": 10, "nh": 50},
        #{"name": "Arch3", "f": 8, "nf": 40, "nh": 50},
        {"name": "Arch2-wide", "f": 4, "nf": 40, "nh": 50},
    ]
    for arch in architectures:
        train_and_eval_net_increaseing_cycle(
            f=arch["f"],
            nf=arch["nf"],
            nh=arch["nh"],
            plot_filename=f"iii_plot{arch['name']}.png"
        )


def do_ex_4_1():
    architectures = [
        {"name": "Arch5_not_smooth", "f": 4, "nf": 40, "nh": 300},
    ]
    for arch in architectures:
        train_and_eval_net_increaseing_cycle(
            f=arch["f"],
            nf=arch["nf"],
            nh=arch["nh"],
            plot_filename=f"iv_plot{arch['name']}.png",
            n_s=800,
            n_cycles=4,
            lam=0.0025
        )

def do_ex_4_2():
    architectures = [
        {"name": "Arch5_smooth_lam=0.002", "f": 4, "nf": 40, "nh": 300},
    ]
    for arch in architectures:
        train_and_eval_net_increaseing_cycle(
            f=arch["f"],
            nf=arch["nf"],
            nh=arch["nh"],
            plot_filename=f"iv_plot{arch['name']}.png",
            n_s=800,
            n_cycles=4,
            lam=0.002,
            use_label_smoothing=True,
            epsilon=0.1
        )