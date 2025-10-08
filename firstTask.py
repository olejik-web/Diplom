import cmath
from enum import Enum, auto
import numpy as np
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D

class Solver:
	def __init__(self, lmda, sigma, gamma):
		self.lmda0 = lmda.real
		self.lmda1 = lmda.imag
		self.sigma0 = sigma.real
		self.sigma1 = sigma.imag
		self.gamma0 = gamma.real
		self.gamma1 = gamma.imag
		self.h = 0.01
		self.delta = 0.01
		self.tauMax = 100
		self.xInterval = (0, 2 * np.pi)
		self.xi1Interval = (-1.2, 1.2)
		self.tauPointsCount = int(self.tauMax / self.delta) + 1
		self.xPointsCount = int(self.xInterval[1] / self.h) + 1
		self.xi1Values = np.zeros((self.tauPointsCount, self.xPointsCount))
		self.xi2Values = np.zeros((self.tauPointsCount, self.xPointsCount))

	def reXiFunction1()
		

	def setInitialCondition(self):
		xAllValues = np.arange(self.xInterval[0], self.xInterval[1] / 2, self.h)
		yAllValues = np.sin(20 * xAllValues) / 40 - 1
		xAllValues = np.concatenate([xAllValues, np.array([xAllValues[-1] + self.h])])
		yAllValues = np.concatenate([yAllValues, np.array([np.sin(20 * (xAllValues[-1])) / 40 + 1])])
		x = np.arange(xAllValues[-1] + self.h, self.xInterval[1], self.h)
		y = np.sin(20 * x) / 40 + 1
		xAllValues = np.concatenate([xAllValues, x])
		yAllValues = np.concatenate([yAllValues, y])
		self.xi1Values[0][:] = yAllValues
	
	def integral(self, tauIndex, functionValues):
		value = 0
		for xIndex in range(int(len(functionValues[tauIndex]))):
			value += functionValues[tauIndex][xIndex] * self.h
		return value

	def g(self, tauIndex, xIndex):
		xi1 = self.xi1Values[tauIndex][xIndex]
		xi2 = self.xi2Values[tauIndex][xIndex]
		return self.lmda1 * xi1 + self.lmda0 * xi2 + self.sigma0 * xi2 * xi1 * xi1 \
			+ self.sigma1 * xi1 * xi1 * xi1 + self.sigma0 * xi2 * xi2 * xi2 \
			+ self.sigma1 * xi1 * xi2 * xi2 + self.gamma0 * self.integral(tauIndex, self.xi2Values) \
			+ self.gamma1 * self.integral(tauIndex, self.xi1Values)

	def f(self, tauIndex, xIndex):
		xi1 = self.xi1Values[tauIndex][xIndex]
		xi2 = self.xi2Values[tauIndex][xIndex]
		return -self.lmda1 * xi2 + self.lmda0 * xi1 + self.sigma0 * xi1 * xi1 * xi1 \
			+ self.sigma1 * xi2 * xi1 * xi1 + self.sigma0 * xi1 * xi2 * xi2 \
			- self.sigma1 * xi1 * xi1 * xi1 + self.gamma0 * self.integral(tauIndex, self.xi1Values) \
			- self.gamma1 * self.integral(tauIndex, self.xi2Values)

	def drawGraph(self):
		plot.figure(figsize=(6, 4))

		plot.subplot(2, 2, 1)
		self.setInitialCondition()
		plot.title(r"$\xi_1$(0, $x$)")
		plot.ylabel(r"$\xi_1$")
		plot.xlabel(r"$x$")
		plot.xlim(self.xInterval[0], self.xInterval[1])
		plot.ylim(self.xi1Interval[0], self.xi1Interval[1])
		plot.grid(True)
		x = np.linspace(self.xInterval[0], self.xInterval[1], len(self.xi1Values[0]))
		plot.plot(x, self.xi1Values[0])

		# calculateCount = 10
		# for tau in range(calculateCount):
		# 	for x in range(self.xPointsCount):
		# 		self.xi2Values[tau + 1][x] = self.xi2Values[tau][x] + self.delta * self.g(tau, x)
		# 		self.xi1Values[tau + 1][x] = self.xi1Values[tau][x] + self.delta * self.f(tau, x)
		# plot.subplot(2, 2, 2)
		# plot.title(r"$\xi_1$({}, $x$)".format(calculateCount * self.delta))
		# plot.ylabel(r"$\xi_1$")
		# plot.xlabel(r"$x$")
		# plot.xlim(self.xInterval[0], self.xInterval[1])
		# plot.ylim(self.xi1Interval[0], self.xi1Interval[1])
		# plot.grid(True)
		# x = np.linspace(self.xInterval[0], self.xInterval[1], self.xPointsCount)
		# plot.plot(x, self.xi1Values[calculateCount])

		# calculateCount = 20
		# for tau in range(calculateCount):
		# 	for x in range(self.xPointsCount):
		# 		self.xi2Values[tau + 1][x] = self.xi2Values[tau][x] + self.delta * self.g(tau, x)
		# 		self.xi1Values[tau + 1][x] = self.xi1Values[tau][x] + self.delta * self.f(tau, x)
		# plot.subplot(2, 2, 3)
		# plot.title(r"$\xi_1$({}, $x$)".format(calculateCount * self.delta))
		# plot.ylabel(r"$\xi_1$")
		# plot.xlabel(r"$x$")
		# plot.xlim(self.xInterval[0], self.xInterval[1])
		# plot.ylim(self.xi1Interval[0], self.xi1Interval[1])
		# plot.grid(True)
		# x = np.linspace(self.xInterval[0], self.xInterval[1], self.xPointsCount)
		# plot.plot(x, self.xi1Values[calculateCount])

		# calculateCount = 30
		# for tau in range(calculateCount):
		# 	for x in range(self.xPointsCount):
		# 		self.xi2Values[tau + 1][x] = self.xi2Values[tau][x] + self.delta * self.g(tau, x)
		# 		self.xi1Values[tau + 1][x] = self.xi1Values[tau][x] + self.delta * self.f(tau, x)
		# plot.subplot(2, 2, 4)
		# plot.title(r"$\xi_1$({}, $x$)".format(calculateCount * self.delta))
		# plot.grid(True)
		# plot.ylabel(r"$\xi_1$")
		# plot.xlabel(r"$x$")
		# plot.xlim(self.xInterval[0], self.xInterval[1])
		# plot.ylim(self.xi1Interval[0], self.xi1Interval[1])
		# x = np.linspace(self.xInterval[0], self.xInterval[1], self.xPointsCount)
		# plot.plot(x, self.xi1Values[calculateCount])

		plot.show()

if __name__ == "__main__":
	Solver(1, 1, -4).drawGraph()
