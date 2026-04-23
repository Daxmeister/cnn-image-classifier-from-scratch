import numpy as np

class Debug_data():
    """Class for handling debug data"""
    
    def __init__(self, debug_file='debug_info.npz'):
        self.load_data = np.load(debug_file)
    
    def load_convolv_gt(self):
        return self.load_data['conv_outputs']
        
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
        
        