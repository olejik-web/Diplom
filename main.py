import cmath
import numpy as np
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D

def drawInitialCondition(plot, xInterval, xiInterval, h, xiValues):
	plot.title(r"$\xi_1$(0, $x$)")
	plot.ylabel(r"$\xi_1$")
	plot.xlabel(r"$x$")
	plot.xlim(xInterval[0], xInterval[1])
	plot.ylim(xiInterval[0], xiInterval[1])
	plot.grid(True)
	x = np.arange(xInterval[0], xInterval[1] / 2, h)
	print("length={}".format(len(x)))
	y = np.sin(20 * x) / 40 - 1
	# plot.plot(x, y)
	# plot.plot([x[-1], x[-1] + h], [y[-1], np.sin(20 * (x[-1] + h)) / 40 + 1], 'tab:blue')
	x = np.concatenate(x, np.array([x[-1] + h]))
	y = np.concatenate(y, np.array([np.sin(20 * (x[-1])) / 40 + 1]))
	x = np.concatenate(x, np.arange(x[-1] + 2 * h, xInterval[1], h))
	y = np.concatenate(y, np.sin(20 * x) / 40 + 1)
	plot.plot(x, y, 'tab:blue')


def drawGraph(lmda, sigma, gamma):
	lmda0 = lmda.real
	lmda1 = lmda.imag
	sigma0 = sigma.real
	sigma1 = sigma.imag
	gamma0 = gamma.real
	gamma1 = gamma.imag
	h = 0.01
	delta = 0.01
	tauMax = 100
	
	xInterval = (0, 2 * np.pi)
	xiInterval = (-1.2, 1.2)
	xiValues = np.zeros((int(tauMax / delta), int(xInterval[1] / h)))
	print("xiValues:", xiValues.shape)

	plot.figure(figsize=(12, 8))

	# plot.subplot(2, 2, 1)
	drawInitialCondition(plot, xInterval, xiInterval, h, xiValues[0])

	#plot.subplot(2, 2, 2)
	#plot.title(r"$\xi_1$(50, $x$)")
	#plot.ylabel(r"$\xi_1$")
	#plot.xlabel(r"$x$")
	#plot.xlim(-1, 1)
	#plot.ylim(-1, 1)
	#plot.grid(True)

	#plot.subplot(2, 2, 3)
	#plot.title(r"$\xi_1$(100, $x$)")
	#plot.ylabel(r"$\xi_1$")
	#plot.xlabel(r"$x$")
	#plot.xlim(-1, 1)
	#plot.ylim(-1, 1)
	#plot.grid(True)

	#plot.subplot(2, 2, 4)
	#plot.title(r"$\xi_1$($\tau$, $x$)")
	#plot.grid(True)
	#plot.ylabel(r"$\xi_1$")
	#plot.xlabel(r"$x$")
	#plot.xlim(-1, 1)
	#plot.ylim(-1, 1)

	plot.show()

if __name__ == "__main__":
	drawGraph(1, 1, -4)
