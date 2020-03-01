import numpy as np

class EqSolver:
	"""
	class represents problem func(x) = 0 and methods to numerically solve it.
	init arguments:
		func(x, constArgs = None) - lhs of given problem. Returns float ndarray, x - float ndarray.
		size - size of a problem( equations and variables count)
		dfunc(x, i, constArgs = None) - represents partial derivative of func by xi. If not provided the numerical derivative is used.Returns float ndarray, x - float ndarray.
		initVal - initial point for numerical methods. If not provided, random vector with vals in range [0, 1) is used.float ndarray.
		
		eps - represents accuracy of numerical methods. positive float
		h - step size, when calculating default derivative. positive float
	
		One can use newton() or broyden() to solve problem using selected method
	"""
	
	def __init__(self, func, size, dfunc = None, constArgs = None, initVal = None, eps = 0.01, h = 0.01, maxiter = 100):
		assert eps > 0.0, 'eps must be positive float'
		assert h > 0.0, 'h must be positive float'
		
		self.func = func
		self.size = size
		if dfunc is None:
			self.dfunc = self.__derivative
		else:
			self.dfunc = dfunc
			
		self.constArgs = constArgs
		if initVal is None:
			self.initVal = np.random.rand(size)
		else:
			self.initVal = initVal
			
		self.eps = eps
		self.h = h
		self.maxiter = maxiter
	
	def solve(self, method):
		if method == 'Newton':
			res = self.__newton()
		elif method == 'Broyden':
			res = self.__broyden()
		else:
			raise ValueError
		
		return res
	
	def __newton(self):
		x = self.initVal
		k = 0
		while(k < self.maxiter):
			dx = self.__linsolve(self.__jacobian(x), - self.func(x, self.constArgs))
			x = x + dx
			if np.dot(dx, dx) < self.eps:
				return x
			else:
				k = k + 1
							
		return None
		
		
	def __broyden(self):
		x = self.initVal
		k = 0
		J = self.__jacobian(x)
		while(k < self.maxiter):
			f0 = self.func(x, self.constArgs)
			dx = self.__linsolve(J, - f0)
			x = x + dx
			if np.dot(dx, dx) < self.eps:
				return x
			else:
				k = k + 1
				y = self.func(x, self.constArgs) - f0
				dxnorm = np.dot(dx, dx)				
				J = J + np.outer((y - np.dot(J, dx)), dx) / dxnorm
		
		return None
		
		
	def __derivative(self, x, n, constArgs = None):
		f0 = self.func(x, self.constArgs)
		x[n] = x[n] + self.h
		f1 = self.func(x, self.constArgs)
		return (f1 - f0) / self.h
		
	def __jacobian(self, x):
		res = np.zeros((self.size, self.size))
		for i in range(self.size):
			res[:, i] = self.dfunc(x, i, self.constArgs)
			
		return res
		
	def __linsolve(self, matrix, rhs):
		res = None
		try:
			res = np.linalg.solve(matrix, rhs)
		except np.linalg.LinAlgError:
			print("solve failed")
			try:
				res,_ ,_ ,_ = np.linalg.lstsq(matrix, rhs, rcond=None)
			except np.linalg.LinAlgError:
				print("lstsq failed")
				raise
			
			
		return res

	
		