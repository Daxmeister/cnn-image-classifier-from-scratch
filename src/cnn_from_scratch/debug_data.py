import numpy as np

class Debug_data():
    """Class for handling debug data"""
    
    def __init__(self, debug_file='debug_info.npz'):
        """
        Loads images and reshapes X into X_ims. 
    
        Stores as:
            self.X_ims: ndarray of shape (32, 32, 3, 5) w/ images in correct shape 
            self.Fs: nd array of shape (4, 4, 3, 2) w/ Filters    
        """
        
        
        self.load_data = np.load(debug_file)
        
        X = self.load_data['X']
        self.Fs = self.load_data['Fs']
        n = X.shape[1]
        self.X_ims = np.transpose(X.reshape((32, 32, 3, n), order='F'), (1, 0, 2, 3))

    def compare_conv_out_to_ground_truth(self, convolver):
        ground_truth= self.load_data['conv_outputs']
        conv_output = convolver.conv_for_loop(self.X_ims, self.Fs)
        self._print_comparison(ground_truth, conv_output)
        
    def compare_for_loop_w_matmul_convolver(self, convolver):
        conv_output_fl = convolver.conv_for_loop(self.X_ims, self.Fs)
        conv_output_matmul = convolver.conv_mat_mul(self.X_ims, self.Fs)
        # Change shape to enable comparison
        conv_output_fl_flat = conv_output_fl.reshape((conv_output_matmul.shape), order='C') # ERROR TODO
        
        self._print_comparison(conv_output_fl_flat, conv_output_matmul)
    
    def _print_comparison(self, ground_truth, conv_output):
        print("Shape convoler is:", conv_output.shape, 
              " Should be: ", ground_truth.shape, 
              "\nShape same? ", conv_output.shape==ground_truth.shape)
        print("\nDifference in values: ",np.sum(np.abs(conv_output-ground_truth)))
        
    def load_X_Fs(self):
        X = self.load_data['X']
        Fs = self.load_data['Fs']
        n = X.shape[1]
        X_ims = np.transpose(X.reshape((32, 32, 3, n), order='F'), (1, 0, 2, 3))
        return X_ims, Fs
    
    def load_W_b(self):
        W1 = self.load_data['W1']
        W2 = self.load_data['W2']
        b1 = self.load_data['b1']
        b2 = self.load_data['b2']
        return [W1, W2], [b1, b2]
    
    def load_fp_output(self):
        X1 = self.load_data['X1'].squeeze(0) # Correct wrong shape in debug X1 data values
        return self.load_data['conv_flat'], X1, self.load_data['P']
    
    def load_backprop_data(self):
        return self.load_data['Y'], self.load_data['grad_Fs_flat']
        
        