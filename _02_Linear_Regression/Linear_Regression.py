# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    x, y = read_data()

    alpha = 1

    w = np.linalg.inv((x.T @ x + alpha * np.eye(np.shape((x.T @ x))[0]))) @ x.T @ y

    return w @ data
    
def lasso(data):
    x, y = read_data()

    alpha = 0.01
    epochs = 10000
    Lambda = 0.001

    w = np.zeros(x.shape[1])

    for i in range(epochs):
        dw = x.T @ ((x @ w) - y.T) + alpha * np.sign(w)
        dw_norm = np.linalg.norm(dw)
        if dw_norm > 1:
            dw /= dw_norm
        w = w - Lambda * dw

    return w @ data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
