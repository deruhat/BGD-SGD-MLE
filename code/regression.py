# Abdulellah Abualshour
# Regression

import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split

# Batch Gradeint Descent
def BGD():
	# Generate data
	f = 1/1000
	sample = 1000
	x_points = np.linspace(0, 3, num=1000)
	noise = 0.0001*np.asarray(random.sample(range(0,1000),sample))
	y_points = np.cos(x_points) + noise
	plt.figure(0)
	plt.xlabel('Variable (x)')
	plt.ylabel('Target (t)')

	# create training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(x_points, y_points, test_size=0.2)
	print(X_train.shape, y_train.shape)
	print(X_test.shape, y_test.shape)

	# plot desired points
	plt.scatter(X_test, y_test, color='blue')

	# print important info
	m = len(X_train)
	print('# of samples:', m)
	max_iterations = 1000
	print('# of max iterations:', max_iterations)
	learning_rate = 1e-1
	print('learning rate:', learning_rate)

	# we define the basis function and y_hat
	# y = w_0 + w_1*x_i
	w = [random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)]
	y_hat = lambda x: w[2]*np.exp(-(x-2)**2) + w[1]*np.exp(-(x-1)**2) + w[0]*np.exp(-x*x) # lambda expression for future use, represents the line equation

	# now we define the function for calculating the sum:
	def sigma(x, y, lambda_expression):
		w_0_total = np.sum((y_hat(x) - y)*np.exp(-x*x))
		w_1_total = np.sum(((y_hat(x) - y)*np.exp(-(x-1)**2)))
		w_2_total = np.sum(((y_hat(x) - y)*np.exp(-(x-2)**2)))
		return w_0_total, w_1_total, w_2_total

	cost = []
	# learning loop (iterations)
	for j in range(max_iterations):
		sum_0, sum_1, sum_2 = sigma(X_train, y_train, y_hat)

		# divide by m
		sum_0 = sum_0 / m
		sum_1 = sum_1 / m
		sum_2 = sum_2 / m

		# learn coefficients
		w[0] = w[0] - learning_rate*sum_0
		w[1] = w[1] - learning_rate*sum_1
		w[2] = w[2] - learning_rate*sum_2

		# append cost function
		cost.append(np.square(np.linalg.norm(y_hat(X_train)-y_train)))

	print(w[0])
	print(w[1])
	print(w[2])

	y_predicted = [y_hat(x) for x in X_test]
	y_predicted = np.array(y_predicted)

	ind = np.argsort(X_test)
	ind = ind.ravel()
	plt.title('Predicted Target (t) on testing data')
	plt.plot(X_test[ind], y_predicted[ind], color='red')

	plt.figure(1)
	n = np.linspace(1, max_iterations, max_iterations)
	plt.plot(n, cost)
	plt.xlabel('Iteration (i)')
	plt.ylabel('Cost (J)')
	plt.title('Const function decrease with iteration increase')
	plt.show()

	cost_test = np.square(np.linalg.norm(y_predicted - y_test))*2/len(X_test)
	print('Root-Mean-Square Error:', cost_test)

# Stochastic Gradient Descent
def SGD():
	# Generate data
	f = 1/1000
	sample = 1000
	x_points = np.linspace(0, 3, num=1000)
	noise = 0.0001*np.asarray(random.sample(range(0,1000),sample))
	y_points = np.cos(x_points) + noise
	plt.figure(0)
	plt.xlabel('Variable (x)')
	plt.ylabel('Target (t)')

	# create training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(x_points, y_points, test_size=0.2)
	print(X_train.shape, y_train.shape)
	print(X_test.shape, y_test.shape)

	# plot desired points
	plt.scatter(X_test, y_test, color='blue')

	# print important data
	m = len(X_train)
	print('# of samples:', m)
	max_iterations = 1000
	print('# of max iterations:', max_iterations)
	learning_rate = 1e-1
	print('learning rate:', learning_rate)

	# we define the basis function and y_hat
	# y = w_0 + w_1*x_i
	w = [random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)]
	y_hat = lambda x: w[2]*np.exp(-(x-2)**2) + w[1]*np.exp(-(x-1)**2) + w[0]*np.exp(-x*x) # lambda expression for future use, represents the line equation

	# now we define the function for calculating the sum (I use it here but we dont sum. I pass one value to the function):
	def sigma(x, y, lambda_expression):
		w_0_total = np.sum((y_hat(x) - y)*np.exp(-x*x))
		w_1_total = np.sum(((y_hat(x) - y)*np.exp(-(x-1)**2)))
		w_2_total = np.sum(((y_hat(x) - y)*np.exp(-(x-2)**2)))
		return w_0_total, w_1_total, w_2_total

	cost = []
	# learning loop (iterations)
	for j in range(max_iterations):
		i = np.random.randint(0,m,1)

		sum_0, sum_1, sum_2 = sigma(X_train[i], y_train[i], y_hat)

		# learn coefficients
		w[0] = w[0] - learning_rate*sum_0
		w[1] = w[1] - learning_rate*sum_1
		w[2] = w[2] - learning_rate*sum_2

		# append cost function
		cost.append(np.square(np.linalg.norm(y_hat(X_train)-y_train)))

	print(w[0])
	print(w[1])
	print(w[2])

	y_predicted = [y_hat(x) for x in X_test]
	y_predicted = np.array(y_predicted)

	ind = np.argsort(X_test)
	ind = ind.ravel()
	plt.title('Predicted Target (t) on testing data')
	plt.plot(X_test[ind], y_predicted[ind], color='red')

	plt.figure(1)
	n = np.linspace(1, max_iterations, max_iterations)
	plt.plot(n, cost)
	plt.xlabel('Iteration (i)')
	plt.ylabel('Cost (J)')
	plt.title('Const function decrease with iteration increase')
	plt.show()

	cost_test = np.square(np.linalg.norm(y_predicted - y_test))*2/len(X_test)
	print('Root-Mean-Square Error:', cost_test)

# Maximum Likeliihood Estimation
def MLE():
	# Generate data
	f = 1/1000
	sample = 1000
	x_points = np.linspace(0, 3, num=1000)
	noise = 0.0001*np.asarray(random.sample(range(0,1000),sample))
	y_points = np.cos(x_points) + noise
	plt.figure(0)
	plt.xlabel('Variable (x)')
	plt.ylabel('Target (t)')

	# create training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(x_points, y_points, test_size=0.2)
	print(X_train.shape, y_train.shape)
	print(X_test.shape, y_test.shape)

	# plot desired data
	plt.scatter(X_test, y_test, color='blue')

	# print important info
	m = len(X_train)
	print('# of samples:', m)
	max_iterations = 1000
	print('# of max iterations:', max_iterations)
	learning_rate = 1e-1
	print('learning rate:', learning_rate)

	# we define the basis function and y_hat
	w = [random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)]
	y_hat = lambda x: w[2]*np.exp(-(x-2)**2) + w[1]*np.exp(-(x-1)**2) + w[0]*np.exp(-x*x) # lambda expression for future use, represents the line equation

	# we build the P matrix (big phai)
	p_0 = np.exp(-X_train*X_train)
	p_0 = np.reshape(p_0, (-1, 1))
	p_1 = np.exp(-(X_train-1)**2)
	p_1 = np.reshape(p_1, (-1, 1))
	p_2 = np.exp(-(X_train-2)**2)
	p_2 = np.reshape(p_2, (-1, 1))
	P = np.hstack((p_0, p_1, p_2))
	print(P.shape)

	# we do the linear algebra to find the weights
	w = np.matmul(np.matmul(np.linalg.inv(np.matmul(P.T, P)), P.T), y_train)

	# cost function
	cost = np.square(np.linalg.norm(y_hat(X_train)-y_train))

	print(w[0])
	print(w[1])
	print(w[2])

	y_predicted = [y_hat(x) for x in X_test]
	y_predicted = np.array(y_predicted)

	ind = np.argsort(X_test)
	ind = ind.ravel()
	plt.title('Predicted Target (t) on testing data')
	plt.plot(X_test[ind], y_predicted[ind], color='red')
	plt.show()
	cost_test = np.square(np.linalg.norm(y_predicted - y_test))*2/len(X_test)
	print('Root-Mean-Square Error:', cost_test)

if __name__ == '__main__':
	BGD()