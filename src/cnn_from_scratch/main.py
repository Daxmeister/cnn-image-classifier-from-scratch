import numpy as np
import debug_data
import convolver

debug_data = debug_data.Debug_data()
for_loop_convolver = convolver.Convolver()
debug_data.compare_conv_out_to_ground_truth(for_loop_convolver)
