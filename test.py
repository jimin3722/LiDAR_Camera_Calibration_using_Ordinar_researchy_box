import numpy as np 
import copy

a = np.array([[4, 5], [ 2, 3], [2, 2]])

b = a / a[2, :]


print (b)