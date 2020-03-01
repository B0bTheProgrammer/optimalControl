import numpy as np

class integralSolver():
	"""
	func(t, x, u)
	"""
	def __init__(self, func, size,  timeVals, xVals, uVals):
		self.func = func
		self.size = size
		self.time = timeVals
		self.x = xVals
		self.u = uVals
		
	def solve(self):
		res = np.zeros(self.size)
		for i in range(self.time.size - 1):
			res = res + (self.func(self.time[i+1], self.x[:, i+1], self.u[:, i+1]) + self.func(self.time[i], self.x[:, i], self.u[:, i]) )/2.0 * (self.time[i+1] - self.time[i])
		
		return res