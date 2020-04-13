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



	features = []
	for i in range(30):
		if i < 15:
			features.append([0, X_en[i][0], X_en[i][1], y_en[i][0]])
		else:
			features.append([1, X_fr[i-15][0], X_fr[i-15][1], y_fr[i-15][0]])
	
	features = np.array(features)
	w = test_classification(features, normalize1)
	






	# x = np.linspace(0,10,1000)
	# k = w[2]/w[1]
	# m = -w[0]/w[1]


	#PLOTS
	# plt.plot(x,k*x+m)

	# plt.plot(features[0:14,1],features[0:14,2], '*')
	# plt.plot(features[15:29,1],features[15:29,2], 'ro')
	# plt.show()
	

def test_classification(X, normalize1):
	correct = 0

	for i in range(len(X)):
		w_init = np.array([1.0,1.0,1.0])
		train_Xy, test_Xy = split_X(X, i)
		w_new = perceptron_learning(train_Xy, w_init)
		correct += test(test_Xy, w_new)
	print(correct[0], ' / ', len(X))

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
	if weights @ test_Xy[1:] >= 0:
		return test_Xy[0] == 1
	return test_Xy == 0

	
def split_X(X, i):
	test_Xy = X[i][:]
	train_Xy = np.delete(X,i,0)
	return train_Xy, test_Xy



def perceptron_learning(Xy, w):
	ww = w.copy()
	random.seed(0)
	idx = list(range(len(Xy)))

	min_error = 1000

	for i in range(1000):

		
		random.shuffle(idx)
		error = 0
		alpha = 1000/(1000+i)
		for j in range(len(idx)):
			
			y_hat = Threshold(ww @ Xy[j][1:])
			ww[0] += alpha*((Xy[j][0] - y_hat))
			ww[1] += alpha*(Xy[j][2] * (Xy[j][0] - y_hat))
			ww[2] += alpha*(Xy[j][3] * (Xy[j][0] - y_hat))
			error += (Xy[j][0] - y_hat)**2
		if error < 1:
			min_error = error
			return ww
	return ww
		#w = w + X @ y_hat







def Threshold(z):
	return z >= 0



if __name__ == '__main__':
    main()