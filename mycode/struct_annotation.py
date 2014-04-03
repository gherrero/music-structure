import os
import csvio
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

#	parameters
# filename_list = "desc_list.txt"
desc_path   = '../hpcp_ah6_al5_csv/'
res_path    = '../results'
hop_size    = 0.1393
min_len     = 3
m           = 4
kappa       = 0.03
lvert       = 32
lhor        = 0.3
sigmaFactor = 1
tau         = 1
delta       = 0.05		




loadCSV = False;
saveCSV = False;


flist=open('desc_list.txt','r')

desc_file=flist.readline()
np.set_printoptions(threshold=np.nan)

def distance(X):

	# A   = np.random.randn(540, 2)
	# B   = np.random.randn(540, 2)
	# alpha = 0
	# ind   = np.all(distance_matrix(A, B) > alpha, axis=0)
	i=0
	dist=cdist(X,X,'euclidean')





	return dist

def delaycoord(x,m,tau):
	embeddingWindow = (m-1)*tau
	i = 0
	y = []
	y = np.zeros(x.shape[1]*embeddingWindow)
	for row in x[:-embeddingWindow,:]:
		sec = x[i:i+embeddingWindow,:].flatten()
		y = np.vstack((y,sec))
		i = i + 1
	y = y[1:]
	return y

def annotate(desc_file):

	print("file name: " + desc_path + desc_file[:-1])	
	hpcps = np.loadtxt(desc_path + "Beatles_AllYouNeedIsLove_Beatles_1967-MagicalMysteryTour-11.wav.csv", delimiter=',') # read descriptors
	print hpcps.shape
	y = delaycoord(hpcps, m, tau) # apply delay coordinates
	print("delay coord dimensions: ")
	print(y.shape)
	plt.imshow(y.T)
	dist = distance(y)
	print dist.shape
	# plt.imshow(dist)
	plt.show()

if __name__ == "__main__":
	annotate(desc_file)


