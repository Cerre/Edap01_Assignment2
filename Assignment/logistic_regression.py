#Stochastic gradient descent with logistic regression
import random
import vector
import datasets
import matplotlib.pyplot as plt
import numpy as np
import pdb
from gradient_descent_2 import normalize



def main():
	X_en, y_en = datasets.load_tsv(
	'https://raw.githubusercontent.com/pnugues/ilppp/master/programs/ch04/salammbo/salammbo_a_en.tsv')
	X_fr, y_fr = datasets.load_tsv(
	'https://raw.githubusercontent.com/pnugues/ilppp/master/programs/ch04/salammbo/salammbo_a_fr.tsv')
	X_en = np.array(X_en)
	X_fr = np.array(X_fr)
	y_en = np.array([y_en]).T
	y_fr = np.array([y_fr]).T


	normalize1 = False
	if normalize1:
		X_en, X_max = normalize(X_en)
		y_en, y_max = normalize(y_en)

		X_fr, X_max = normalize(X_fr)
		y_fr, y_max = normalize(y_fr)



	w_init = np.array([10.0,10.0,10.0])
	features = []
	for i in range(30):
		if i < 15:
			features.append([0, X_en[i][0], X_en[i][1], y_en[i][0]])
		else:
			features.append([1, X_fr[i-15][0], X_fr[i-15][1], y_fr[i-15][0]])
	
	features = np.array(features)
	w = test_classification(features, normalize1)



def test_classification(X, normalize1):
	correct = 0

	for i in range(len(X)):
		w_init = np.array([1.0,1.0,1.0])
		train_Xy, test_Xy = split_X(X, i)
		w_new = logistic_regression(train_Xy, w_init)
		correct += test(test_Xy, w_new)
	print(correct, ' / ', len(X))

	k = -w_new[1]/w_new[2]
	m = -w_new[0]/w_new[2]
	print(k,m)
	if normalize1: 
		x = np.linspace(0,1,30)
	else:
		x = np.linspace(0,80000,30)
		
	y1 = k*x + m
	line_1, = plt.plot(x,y1, label = "English Batch")
	plot1, = plt.plot(X[0:14,2],X[0:14,3],'*', label = "English Dataset")
	plot2, = plt.plot(X[15:29,2],X[15:29,3],'ro', label = "French Dataset")
	plt.legend(handles = [line_1, plot1, plot2])
	plt.show()


def test(test_Xy, weights):
	if sigmoid(weights @ test_Xy[1:]) >= 0.5:
		return test_Xy[0] == 1
	return test_Xy[0] == 0



def logistic_regression(Xy, w):
	random.seed(0)
	idx = list(range(len(Xy)))
	epsilon = 0.005 #When the change of w is smaller than epsilon, we're happy
	alpha = 0.5
	weights = w.copy()


	for epoch in range(1000):
		random.shuffle(idx)
		for j in idx:
			weights[0] += alpha*(Xy[j][0] - sigmoid(weights @ Xy[j][1:]))
			weights[1] += alpha*Xy[j][2]*(Xy[j][0] - sigmoid(weights @ Xy[j][1:]))
			weights[2] += alpha*Xy[j][3]*(Xy[j][0] - sigmoid(weights @ Xy[j][1:]))

			# if np.linalg.norm(w - w_old) / np.linalg.norm(w) < epsilon:
			# 	print("Number of epochs to convergence in stochastic: ", epoch)
			# 	return w
	return weights



def sigmoid(z):
	return 1 / (1 + np.exp(-z)) >= 0.5





def split_X(X, i):
	test_Xy = X[i][:]
	train_Xy = np.delete(X,i,0)
	return train_Xy, test_Xy


















main()