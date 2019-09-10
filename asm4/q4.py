###################### Assignment 4 ##########################
####################### Question 4 ###########################
# Abhinav Tripathi
# 15807023
# This code solves for eta = 1/t
# For solving other parts, edit line number 29

import numpy as np 
# import cvxpy as cvx 
import matplotlib.pyplot as plt 

if __name__ == '__main__':
    m = 1000
    n = 200

    np.random.seed(0)
    # Choose a and b randomly
    a = np.random.normal(0,1,(n,m))
    b = np.random.normal(0,1,m)

    # Choose initial x randomly
    x = np.random.rand(n)

    f_t_best_cache = []
    epsilon = 1000

    t = 0
    # Until convergence with error of 0.0001 
    while epsilon>1e-2:
        eta = 1/(t+1)
        # Calculate J = aT x + b
        J = np.dot(a.T, x) + b 
        # print(J.shape) (1000,)

        j_t = np.argmax(J)
        g_t = a[:, j_t]
        # print(g_t.shape) 

        # Update x
        x_old = x
        x = x - eta * g_t
        # print('g_t', g_t)
        x[x < 0] = 0
        # print('x_old-x', x_old-x)
        epsilon = np.linalg.norm(x_old-x)
        # print('epsilon', epsilon)

        if t == 0:
            f_t_best = J[j_t]
        
        f_t = J[j_t]
        if f_t_best > f_t:
            f_t_best = f_t
        f_t_best_cache.append(f_t_best)
        t = t+1
        
       

    x_star = x
    f_x_star = J[j_t]

    # print('x_star', x_star)
    # print('f_x_star', f_x_star)
    # print('f_t_best_cache', f_t_best_cache)
    # print('t', t)
    plt.plot(f_t_best_cache - f_x_star)
    plt.show()
    