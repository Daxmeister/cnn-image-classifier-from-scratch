import numpy as np

def compute_loss(P, y): 
    p_correct_classes = P[y, np.arange(len(y))] # nx1 shape
    loss_per_datapoint_vector = -1*np.log(p_correct_classes) # 1xn shape
    return np.average(loss_per_datapoint_vector) # Note, unlike equation 5 no regularization is used

def compute_accuracy(P, y):
    # P is Kxn
    # y is a list (not ndarray) of length n
    predictions = np.argmax(P, axis=0)
    ncorrect = np.sum(predictions==y)
    return ncorrect/len(y)

def softmax(s):
    
    s_shift = s - np.max(s, axis=0, keepdims=True) # Shift to prevent overflow
    s_exp = np.exp(s_shift)
    P = s_exp / np.sum(s_exp, axis=0, keepdims=True) # We broadcast the columnwise sums to get P

    
    """s_exp = np.exp(s) # Note, we do not shift so may overflow
    numerator = 1/np.sum(s_exp, axis=0) # column wise sum nx1
    P = s_exp * numerator """
    return P #shape Kxn

def update_eta(GD_params, t, is_rising):
    dif = GD_params["eta_max"]-GD_params["eta_min"]
    if is_rising:
        return GD_params["eta_min"] + (t)/GD_params["n_s"] * (dif)
    else:
        return GD_params["eta_max"] - (t)/GD_params["n_s"] * (dif)
