###################### Assignment 3 ##########################
####################### Question 2 ###########################
# Abhinav Tripathi
# 15807023
# This piece of code projects a point in R^n to a probability
# simplex.
 
import numpy as np
import time

def find_projection(lambd, lambd_new, mu, mu_new, eps):
    itern = 0
    start = time.time()
    while (np.any(lambd_new - lambd) > eps) or (mu_new - mu > eps):
        mu = mu_new
        lambd = lambd_new
        mu_new = (np.sum(x) + np.sum(lambd) - 1)/n
        lambd_new = mu_new * np.ones((n)) - x
        lambd_new[lambd_new < 0] = 0
        # print('l: ', lambd_new)
        # print('m: ', mu_new)
        itern = itern + 1
    end = time.time()
    print('Total iterations: ', itern)
    print('Time taken in seconds: ', end - start)
    return lambd_new, mu_new



# Choose the parameters:
n = 3
x = np.random.rand((n))*10
print('x: ', x)
if n != len(x):
    print('Error')
    print('The dimension of x must be equal to n')
    exit()

# Initialize lambda and mu randomly
lambd = np.random.rand((n))
mu = np.random.rand((1))
# print(lambd)
# print(mu)

mu_new = (np.sum(x) + np.sum(lambd) - 1)/n
lambd_new = mu_new * np.ones((n)) - x
lambd_new[lambd_new < 0] = 0
# print(lambd_new)
# print(mu_new)

# Repeat until values converge
eps = 0.00001
lambd_new, mu_new = find_projection(lambd, lambd_new, mu, mu_new, eps)

print('Final value of lambda: ', lambd_new)
print('Final value of mu', mu_new)
u = x + lambd_new - mu_new*np.ones(n)  
print('Required projected vector: ', u)  