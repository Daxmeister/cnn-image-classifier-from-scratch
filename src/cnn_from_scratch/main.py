import numpy as np
import debug_data
import convolver

debug_data = debug_data.Debug_data()
convolver_object = convolver.Convolver()

debug_data.compare_conv_out_to_ground_truth(convolver_object)
debug_data.compare_for_loop_w_matmul_convolver(convolver_object)
