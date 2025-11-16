import cmath
from enum import Enum, auto
import numpy as np
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve

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
		self.reXiInterval = (-1.2, 1.2)
		self.tauPointsCount = int(self.tauMax / self.delta) + 1
		self.xPointsCount = int(self.xInterval[1] / self.h) + 1
		self.reXiValues = np.zeros((self.tauPointsCount, self.xPointsCount))
		self.imXiValues = np.zeros((self.tauPointsCount, self.xPointsCount))

	def equations(self, variables):
		x, y = variables
		eq1 = x ** 2 + y ** 2 - 4
		eq2 = x + y + 8
		return [eq1, eq2]

	# def reXiFunction1()

	def function1(self, rho1, rho2, alpha, betta , x, tau, phi):
		multiplier = np.exp(1j * betta * tau)
		if x <= alpha:
			return rho1 * multiplier + np.cos(20 * x) / 100
		else:
			return rho2 * np.exp(phi * 1j) + + np.cos(20 * x) / 100

	def setInitialCondition(self):
		xValues = np.arange(self.xInterval[0], self.xInterval[1], self.h)
		for i, x in enumerate(xValues):
			yValue = self.function1(-1, 1, np.pi, 2, x, 0, 0)
			self.reXiValues[0][i] = np.real(yValue)
			self.imXiValues[0][i] = np.imag(yValue)
	
	def integral(self, tauIndex, functionValues):
		value = 0
		for xIndex in range(int(len(functionValues[tauIndex]))):
			value += functionValues[tauIndex][xIndex] * self.h
		return value

	def g(self, tauIndex, xIndex):
		reXi = self.reXiValues[tauIndex][xIndex]
		imXi = self.imXiValues[tauIndex][xIndex]
		return self.lmda1 * reXi + self.lmda0 * imXi + self.sigma0 * imXi * reXi * reXi \
			+ self.sigma1 * reXi * reXi * reXi + self.sigma0 * imXi * imXi * imXi \
			+ self.sigma1 * reXi * imXi * imXi + self.gamma0 * self.integral(tauIndex, self.imXiValues) \
			+ self.gamma1 * self.integral(tauIndex, self.reXiValues)

	def f(self, tauIndex, xIndex):
		reXi = self.reXiValues[tauIndex][xIndex]
		imXi = self.imXiValues[tauIndex][xIndex]
		return -self.lmda1 * imXi + self.lmda0 * reXi + self.sigma0 * reXi * reXi * reXi \
			+ self.sigma1 * imXi * reXi * reXi + self.sigma0 * reXi * imXi * imXi \
			- self.sigma1 * reXi * reXi * reXi + self.gamma0 * self.integral(tauIndex, self.reXiValues) \
			- self.gamma1 * self.integral(tauIndex, self.imXiValues)

	def drawGraph(self):

		solution = fsolve(self.equations, [1, 1])
		print("Solution to the system:", solution)

		# plot.figure(figsize=(6, 4))

		# plot.subplot(1, 2, 1)
		# self.setInitialCondition()
		# plot.title(r"$\xi_1$(0, $x$)")
		# plot.ylabel(r"$\xi_1$")
		# plot.xlabel(r"$x$")
		# plot.xlim(self.xInterval[0], self.xInterval[1])
		# plot.ylim(self.reXiInterval[0], self.reXiInterval[1])
		# plot.grid(True)
		# x = np.linspace(self.xInterval[0], self.xInterval[1], len(self.reXiValues[0]))
		# plot.plot(x, self.reXiValues[0])

		# # calculateCount = 10
		# # for tau in range(calculateCount):
		# # 	for x in range(self.xPointsCount):
		# # 		self.imXiValues[tau + 1][x] = self.imXiValues[tau][x] + self.delta * self.g(tau, x)
		# # 		self.reXiValues[tau + 1][x] = self.reXiValues[tau][x] + self.delta * self.f(tau, x)
		# # plot.subplot(2, 2, 2)
		# # plot.title(r"$\xi_1$({}, $x$)".format(calculateCount * self.delta))
		# # plot.ylabel(r"$\xi_1$")
		# # plot.xlabel(r"$x$")
		# # plot.xlim(self.xInterval[0], self.xInterval[1])
		# # plot.ylim(self.reXiInterval[0], self.reXiInterval[1])
		# # plot.grid(True)
		# # x = np.linspace(self.xInterval[0], self.xInterval[1], self.xPointsCount)
		# # plot.plot(x, self.reXiValues[calculateCount])

		# # calculateCount = 20
		# # for tau in range(calculateCount):
		# # 	for x in range(self.xPointsCount):
		# # 		self.imXiValues[tau + 1][x] = self.imXiValues[tau][x] + self.delta * self.g(tau, x)
		# # 		self.reXiValues[tau + 1][x] = self.reXiValues[tau][x] + self.delta * self.f(tau, x)
		# # plot.subplot(2, 2, 3)
		# # plot.title(r"$\xi_1$({}, $x$)".format(calculateCount * self.delta))
		# # plot.ylabel(r"$\xi_1$")
		# # plot.xlabel(r"$x$")
		# # plot.xlim(self.xInterval[0], self.xInterval[1])
		# # plot.ylim(self.reXiInterval[0], self.reXiInterval[1])
		# # plot.grid(True)
		# # x = np.linspace(self.xInterval[0], self.xInterval[1], self.xPointsCount)
		# # plot.plot(x, self.reXiValues[calculateCount])

		# calculateCount = 10
		# for tau in range(calculateCount):
		# 	for x in range(self.xPointsCount):
		# 		self.imXiValues[tau + 1][x] = self.imXiValues[tau][x] + self.delta * self.g(tau, x)
		# 		self.reXiValues[tau + 1][x] = self.reXiValues[tau][x] + self.delta * self.f(tau, x)
		# plot.subplot(1, 2, 2)
		# plot.title(r"$\xi_1$({}, $x$)".format(calculateCount * self.delta))
		# plot.grid(True)
		# plot.ylabel(r"$\xi_1$")
		# plot.xlabel(r"$x$")
		# plot.xlim(self.xInterval[0], self.xInterval[1])
		# plot.ylim(self.reXiInterval[0], self.reXiInterval[1])
		# x = np.linspace(self.xInterval[0], self.xInterval[1], self.xPointsCount)
		# plot.plot(x, self.reXiValues[calculateCount])
		# print(self.imXiValues[calculateCount][:10])

		# plot.show()

if __name__ == "__main__":
	Solver(1, 1, -4).drawGraph()
