import random
#import vector
import datasets
import matplotlib.pyplot as plt
import numpy as np
import pdb


def main():
	X_en, y_en = datasets.load_tsv(
		'https://raw.githubusercontent.com/pnugues/ilppp/master/programs/ch04/salammbo/salammbo_a_en.tsv')
	X_fr, y_fr = datasets.load_tsv(
		'https://raw.githubusercontent.com/pnugues/ilppp/master/programs/ch04/salammbo/salammbo_a_fr.tsv')


	X_en = np.array(X_en)
	X_fr = np.array(X_fr)
	y_en = np.array([y_en]).T
	y_fr = np.array([y_fr]).T
	X_org_en = X_en
	X_org_fr = X_fr
	y_org_en = y_en
	y_org_fr = y_fr

	


	alpha = 1.0e-11
	normalize1 = True
	maxima_en = 0
	maxima_fr = 0
	w = np.zeros(X_en.shape[1]).reshape((-1, 1))
	print(w)


	#English Salammbo
	#Normalize the vectors X and y
	if normalize1:
		X_en, X_max = normalize(X_en)
		y_en, y_max = normalize(y_en)
		maxima_en = np.concatenate((X_max, y_max))
		maxima_en = maxima_en.reshape(-1, 1)
		alpha = 1

	
	
	

	#Stochastic  English
	w_en_s = stochastic(X_en, y_en,alpha,w)
	print("ENGLISH SALAMMBO", '\n')
	print("Stochastic weights: ", w_en_s)
	if normalize1:
		w_en_s = maxima_en[-1, 0] * (w_en_s / maxima_en[:-1, 0:1])
		print("Restored stochastic weights", w_en_s)

	#Batch English
	w_en_b = batch(X_en, y_en,alpha,w)
	print("Batch weights: ", w_en_b)
	if normalize1:
		w_en_b = maxima_en[-1, 0] * (w_en_b / maxima_en[:-1, 0:1])
		print("Restored batch weights", w_en_b)
		print('\n')


	


	#French Salammbo
	if normalize1:
		X_fr, X_max = normalize(X_fr)
		y_fr, y_max = normalize(y_fr)
		maxima_fr = np.concatenate((X_max, y_max))
		alpha = 1
		maxima_fr = maxima_fr.reshape(-1, 1)




	w_fr_s = stochastic(X_fr, y_fr,alpha,w)
	print("FRENCH SALAMMBO", '\n')
	print("Stochastic weights: ", w_fr_s)
	if normalize1:
		w_fr_s = maxima_fr[-1, 0] * (w_fr_s / maxima_fr[:-1, 0:1])
		print("Restored stochastic weights", w_fr_s)


	w_fr_b = batch(X_fr, y_fr,alpha,w)
	print("Batch weights: ", w_fr_b)
	if normalize1:
		w_fr_b = maxima_fr[-1, 0] * (w_fr_b / maxima_fr[:-1, 0:1])
		print("Restored batch weights", w_fr_b)

	#PLOT________________________
	x = np.linspace(0,80000,100)
	y1 = w_en_b[1]*x + w_en_b[0]
	y2 = w_fr_b[1]*x + w_fr_b[0]
	y3 = w_en_s[1]*x + w_en_s[0]
	y4 = w_fr_s[1]*x + w_fr_s[0]
	plt.plot(X_org_en[:,1], y_org_en,'bs')
	plt.plot(X_org_fr[:,1], y_org_fr,'ro')

	line_1, = plt.plot(x,y1, label = "English Batch")
	line_2, = plt.plot(x,y2, label = "French Batch")
	line_3, = plt.plot(x,y3, label = "English Stochastic")
	line_4, = plt.plot(x,y4, label = "French Stochastic")
	plt.legend(handles = [line_1, line_2, line_3, line_4])
	plt.show()




def stochastic(X,y,alpha,w):
	random.seed(0)
	idx = list(range(len(X)))
	epsilon = 0.000005 #When the change of w is smaller than epsilon, we're happy


	for epoch in range(1000):
		random.shuffle(idx)
		for i in idx:
			w_old = w
			w = w + alpha*(y[i] - X[i] @ w) * X[i].reshape(-1, 1)
			if np.linalg.norm(w - w_old) / np.linalg.norm(w) < epsilon:
				print("Number of epochs to convergence in stochastic: ", epoch)
				return w
	return w






def batch(X, y, alpha, w):
	q = len(X)
	random.seed(0)
	idx = list(range(len(X)))
	random.shuffle(idx)
	epsilon = 0.000005 #When the change of w is smaller than epsilon, we're happy

	alpha /= q
	for epoch in range(1000):
		loss = y - X @ w
		gradient = X.T @ loss
		w_old = w
		w = w + alpha * gradient
		if np.linalg.norm(w - w_old) / np.linalg.norm(w) < epsilon:
			print("Number of epochs to convergence in batch: ", epoch)
			return w	
	return w
	




def unnormalize(w0,w1,maxima):
	return w0*maxima, w1*maxima




def loss(w,X,y):
	err = y - X @ w
	#print(err.T @ err)
	return err.T @ err


def normalize(X):
	maxima = np.amax(X, axis=0)
	D = np.diag(maxima)
	D_inv = np.linalg.inv(D)
	X = X @ D_inv
	return (X, maxima)








if __name__ == '__main__':
    main()