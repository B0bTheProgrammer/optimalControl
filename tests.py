import num_methods
import numpy as np
import math
import matplotlib.pyplot as plt

outputSz = 2
W = np.random.random((outputSz,outputSz))
#W = np.array([[1.0,2.0],[3.0,4.0]])
input = np.array([1.0,2.0])
#print(W)
def func(x, constArgs = None):
	return np.dot(W,x)

	
#print(func(input))
solver = num_methods.equations.EqSolver(func, outputSz, initVal = np.array([10.0,10.0]))

#print(solver.func(input))

#print(solver.dfunc(input,1))

#tmp = solver._EqSolver__jacobian(input)
#print(tmp)

print(solver.solve('Newton'))

print(solver.solve('Broyden'))
"""
tmp = solver._EqSolver__linsolve(np.array([[1,2], [3,4]]), np.array([1,1]))
print(tmp)
tmp = solver._EqSolver__linsolve(np.array([[6.0,2], [3,1]]), np.array([1,1]))
print(tmp)
"""

#solver = num_methods.equations.EqSolver(func, outputSz, eps = -2.0)
def ode(t, x, u):
	#return math.cos(t)
	return x

s = num_methods.ode.ODESolver(ode, np.array([10.0]), np.array([-5.0, 5.0]))
t ,x = s.solve('FinitDiff')
plt.plot(t, x[0,:])
plt.show()

def intfunc(t, x, u):
	pass

#s2 = num_methods.integralSolver(t, x)
res = s2.solve()