import numpy as np

class ODESolver():
	"""
	constArgs - V
	ode(t, x, u, constArgs = None) 
	ics - initial conditions (float dnarray, size same as ode output)
	ufunc(t, x, constArgs = None)
	t - time interval [t0, t1]
	h - step size (for fixed step methods)
	
	eps - accuracy for adaptive size methods
	
	"""
	def __init__(self, ode, ics, t, ufunc = None, usize = None, constArgs = None,  h = 0.01, eps = 0.01):
		assert eps > 0.0, 'eps must be positive float'
		assert h > 0.0, 'h must be positive float'
		#assert ufunc != None and usize == None, 'provide input size for ufunc'
		
		self.ode = ode
		self.ics = ics
		if ufunc == None:
			self.ufunc = self.__ufuncDefault
			self.usize = 1
		else:
			self.ufunc = ufunc 
			self.usize = usize
		
		self.constArgs = constArgs		
		self.t = t
		self.h = h
		self.eps = eps
	
	def solve(self, method):
		if method == 'Runge-Kutta':
			res = self.__runge_kutta()
		elif method == 'FinitDiff':
			res = self.__finit_diff()
		else: 
			raise ValueError
		
		return res
		
	def __runge_kutta(self):
		raise NotImplementedError
	
	def __ufuncDefault(self, t, x):
		return 0
	
	def __finit_diff(self):
		t0, t1 = self.t
		sz = int((t1 - t0) / self.h + 1)
		res = np.zeros((self.ics.size, sz))
		u = np.zeros( (self.usize, sz) )
		time = np.zeros(sz)
		time[0] = t0
		res[:, 0] = self.ics
		u[:, 0] = self.ufunc(t0, self.ics, self.constArgs)
		for i in range(1, sz):
			res[:, i] = res[:, i-1] + self.h*self.ode(self.h * (i-1), res[:, i-1], u[:, i-1], self.constArgs)
			time[i] = self.h * i
			u[:, i] = self.ufunc(time[i], res[:, i], self.constArgs)
			
		return time, res, u
		

	
