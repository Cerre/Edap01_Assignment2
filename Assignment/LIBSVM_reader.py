import random
import vector
import datasets
import matplotlib.pyplot as plt
import numpy as np
import pdb
from gradient_descent_2 import normalize

X_en, y_en = datasets.load_tsv(
	'https://raw.githubusercontent.com/pnugues/ilppp/master/programs/ch04/salammbo/salammbo_a_en.tsv')
X_fr, y_fr = datasets.load_tsv(
	'https://raw.githubusercontent.com/pnugues/ilppp/master/programs/ch04/salammbo/salammbo_a_fr.tsv')


X_en = np.array(X_en)
X_fr = np.array(X_fr)
y_en = np.array([y_en]).T
y_fr = np.array([y_fr]).T
X_en, X_max = normalize(X_en)
y_en, y_max = normalize(y_en)
X_fr, X_max = normalize(X_fr)
y_fr, y_max = normalize(y_fr)


f = open("SVM_format.txt","w+")

text = ""

for i in range(15):
	text += "0 "
	text += "1:"
	text += str(X_en[i][0])
	text += " "
	text += "2:"
	text += str(X_en[i][1])
	text += " "
	text += "3:"
	text += str(y_en[i][0])
	text += '\n'
for i in range(15):
	text += "1 "
	text += "1:"
	text += str(X_fr[i][0])
	text += " "
	text += "2:"
	text += str(X_fr[i][1])
	text += " "
	text += "3:"
	text += str(y_fr[i][0])
	text += '\n'
text.strip('\n')

print(text)
f.write(text)
f.close()



