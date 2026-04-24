from cnn_from_scratch import data_handling


def test_load_all_data():
    """ Tests that all data can be read and that validation data size is correct"""
    n_validation = 1000
    X_tr, Y_tr, y_tr, X_val, Y_val, y_val, X_te, Y_te, y_te  = data_handling.load_all_data(n_validation)
    
    assert X_tr.shape[1] + n_validation == 50000
    assert X_val.shape[1] == n_validation

def test_pre_processing():
    """ Tests that pre-processing does not throw errors"""
    n_validation = 1000
    X_tr, Y_tr, y_tr, X_val, Y_val, y_val, X_te, Y_te, y_te  = data_handling.load_some_data(n_validation)
    
    X_tr_c, X_val_c, X_te_c = data_handling.pre_processing(X_tr, X_val, X_te)


    
def test_construct_MX():
    
    n_validation = 1000
    X_tr, Y_tr, y_tr, X_val, Y_val, y_val, X_te, Y_te, y_te  = data_handling.load_some_data(n_validation)
    print(X_tr.shape) # (3072, 9000)
    f = 2
    d = 3
    n_filter_pixels = d*f*f
    n_p = X_tr.shape[0]/(n_filter_pixels)
    n = X_tr.shape[1] 
    
    MX = data_handling.construct_MX(X_tr, f, d)
    assert MX.shape == (n_p, n_filter_pixels, n)