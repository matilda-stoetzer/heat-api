# tester för main.py
import pytest
import numpy as np
from heat import HeatObj

f = lambda x: np.sin(2*np.pi*x)**2
l = lambda t: 0*t
r = lambda t: 0*t
x_int = np.array([0, 1])
t_int = np.array([0, 0.5])
test_object = HeatObj(f, l, r, 10, 100, x_int, t_int, D=1)

#print(test_object.create_solution_matrix().shape)
print(test_object.n)
test_object._update_step_size()
print(test_object.h)
print(test_object.k)


