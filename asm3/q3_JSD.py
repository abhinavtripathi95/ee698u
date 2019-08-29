###################### Assignment 3 ##########################
####################### Question 3 ###########################
# Abhinav Tripathi
# 15807023
# 
# 

import numpy as np
import time
from scipy.spatial.distance import jensenshannon  as JSD

def find_projection(n, x_tbp):
    lambd = np.random.rand((n))
    mu = np.random.rand((1))

    mu_new = (np.sum(x) + np.sum(lambd) - 1)/n
    lambd_new = mu_new * np.ones((n)) - x
    lambd_new[lambd_new < 0] = 0

    # Repeat until values converge
    eps = 0.00001

    itern = 0
    start = time.time()
    while (np.any(lambd_new - lambd) > eps) or (mu_new - mu > eps):
        mu = mu_new
        lambd = lambd_new
        mu_new = (np.sum(x) + np.sum(lambd) - 1)/n
        lambd_new = mu_new * np.ones((n)) - x_tbp
        lambd_new[lambd_new < 0] = 0
        # print('l: ', lambd_new)
        # print('m: ', mu_new)
        itern = itern + 1
    end = time.time()
    u = x_tbp + lambd_new - mu_new*np.ones(n)
    return u

def get_f_x(x, x0, y):
    f_x = (np.linalg.norm(x - x0))**2 + (JSD(x,y))**2
    return f_x


if __name__ == '__main__':

    # Choose the parameters
    n = 100
    eta = (1/n)/100
    x0 = np.random.rand((n)) * 10
    y = np.random.rand((n)) * 10

    # Start projected gradient descent from 1
    x = np.ones((n))
    f_x = get_f_x(x, x0, y)

    grad_f_x = 2*(x - x0) + 0.5 * np.log(2*x/(x+y))

    # The value of x to be projected
    x_tbp = x - eta * grad_f_x
    x_new = find_projection(n, x_tbp)

    while np.any(x_new - x) > 0.00001:
        x_old = x
        x = x_new
        # print(x)
        x[x <= 0] = 1 # redefine the terms where log takes an argument of 0
        grad_f_x = 2*(x - x0) + 0.5 * np.log(2*x/(x+y))
        x_tbp = x - eta * grad_f_x
        x_new = find_projection(n, x_tbp)
        
        if (x_old == x_new).all:
            print('The value is oscillating')
            break
    print('The value is oscillating between: ', x)
    print('and: ', x_new)


