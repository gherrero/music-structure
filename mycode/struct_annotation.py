import os
import csvio
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy import spatial
from scipy import signal

from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors as nn
import matplotlib.cm as cm
import cProfile
from collections import deque

#	parameters
# filename_list = "desc_list.txt"
desc_path   = '../hpcp_ah6_al5_csv/'
res_path    = '../results'
hop_size    = 0.1393
min_len     = 3
m           = 25
kappa       = 0.03
lver        = 32
lhor        = 0.3
sigmaFactor = 1
tau         = 1
delta       = 0.05		




loadCSV = False;
saveCSV = False;


flist=open('desc_list.txt','r')

desc_file=flist.readline()
np.set_printoptions(threshold=np.nan)


# ----profiling scripts-----

def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiled_func

# --------------------------

def ssm(X,kappa):
	K=int(kappa*X.shape[0])
	print "K = "+str(K)
	R=np.zeros([X.shape[0],X.shape[0]])
	i=0
	# tree=spatial.KDTree(X,leafsize=10) #using scipy
	tree=neighbors.KDTree(X,leaf_size=100,p=2.0) #using sklearn
	for row in X:
		# distance, index = tree.query(row,k=K,p=2)
		index=tree.query(X[i,:],k=K,return_distance=False)
		R[i,index]=1
		i=i+1
	RT=R.T
	R=R*RT
	return R

# def delaycoord(x,m,tau):
# 	embeddingWindow = (m-1)*tau
# 	i = 0
# 	y = np.zeros(x.shape[1]*embeddingWindow)
# 	newx=x[:-embeddingWindow,:] #slice it outside the loop so it doesn't create a copy in every iteration
# 	for i, row in enumerate(newx):
# 		sec = x[i:i+embeddingWindow,:].flatten()
# 		y = np.vstack((y,sec))
# 	y = y[1:]
# 	return y

# DELAY COORD JOAN (slightly faster)
def delaycoord(x,m,tau):
	y=np.zeros((x.shape[0]-(m-1)*tau,x.shape[1]*m))
	for j in range(0,m):
		posj=j*x.shape[1]
		posi=j*tau
		y[:,posj:posj+x.shape[1]]=x[posi:posi+y.shape[0],:]
	return y

# def rolling(a, window):
#     shape = (a.size - window + 1, window)
#     strides = (a.itemsize, a.itemsize)
#     return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def circshift(x):
	y=np.zeros((x.shape))
	for i, row in enumerate(x):
		x=np.roll(row,-i)
		y[i]=x
	return y.T

def gaussianblur(x,lhor,lver,std):
	w=signal.gaussian


# @do_cprofile
def annotate(desc_file):

	print("file name: " + desc_path + desc_file[:-1])	
	hpcps = np.loadtxt(desc_path + "Beatles_AllYouNeedIsLove_Beatles_1967-MagicalMysteryTour-11.wav.csv", delimiter=',') # read descriptors
	print "length hpcps: "+str(hpcps.shape[0])
	# plt.imshow(hpcps.T,interpolation='nearest',aspect='auto')
	y = delaycoord(hpcps, m, tau) # apply delay coordinates
	print("delay coord dimensions: ")+ str(y.shape)
	plt.figure()
	R = ssm(y,kappa)
	plt.imshow(R,cmap = cm.binary,interpolation='nearest')
	L=circshift(R)
	plt.figure()
	plt.imshow(L,cmap = cm.binary,interpolation='nearest',aspect='auto')

	plt.show()



if __name__ == "__main__":
	annotate(desc_file)


