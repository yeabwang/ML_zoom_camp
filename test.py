import numpy as np
from typing import List, Tuple

def cost_function(x,y,w,b):
    m = len(x)
    prediction = np.dot(x,w) + b

    cost = np.sum((prediction - y) **2) / (2 * m)
    return cost

def gradient(x,y,w,b):
    m = len(x)
    prediction = np.dot(x,w) + b
    dw = np.sum((prediction - y) * x)/m
    db = np.sum(prediction - y)/m

    return dw, db

def gradient_decent(x,y,alpha,num_iters, tolerance):
    w,b = 0,0
    all_costs = []

    for i in range(num_iters):
        dw,db = gradient(x,y,w,b)

        dw -= alpha * dw
        db -= alpha * db

        cost = cost_function(x,y,w,b)
        all_costs.append(cost)

        if i > 0 and abs(all_costs[i-1] - all_costs[i]) < tolerance:
            break
    
    return w,b, all_costs



def run_lin_reg(x,y, r = 0.01):

    ones = np.ones(x.shape[0])
    x = np.column_stack((ones, x))
    xtx = x.T.dot(x)
    xtx += r * np.eye(xtx.shape[0])
    xtx_inv = np.linalg.inv(xtx)
    w_full = xtx_inv.dot(x.T).dot(y)

