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

hop_size        = 0.1393
min_len         = 3
m               = 22
kappa           = 0.03
lver            = 215
lhor            = 2
sigmaFactor     = 1
tau             = 1
delta           = 0.05		
n               = 100



# flist=open(desc_list,'r')

# desc_file=flist.readline()
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
	K = int(kappa*X.shape[0])
	# print "K = "+str(K)
	R = np.zeros([X.shape[0],X.shape[0]])
	i = 0
	# tree=spatial.KDTree(X,leafsize=10) #using scipy
	tree=neighbors.KDTree(X,leaf_size=100,p=2.0) #using sklearn
	for i in range(X.shape[0]):
		# distance, index = tree.query(row,k=K,p=2)
		index = tree.query(X[i,:],k=K,return_distance=False)
		R[i,index] = 1
		# i=i+1
	RT = R.T
	R = R*RT
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
def delayCoord(x,m,tau):
	y = np.zeros((x.shape[0]-(m-1)*tau,x.shape[1]*m))
	for j in range(0,m):
		posj = j*x.shape[1]
		posi = j*tau
		y[:,posj:posj+x.shape[1]] = x[posi:posi+y.shape[0],:]
	return y

def circShift(x):
	y = np.zeros((x.shape))
	for i, row in enumerate(x):
		x = np.roll(row,-i)
		y[i] = x
	return y.T

def gaussianBlur(x,lhor,lver,std):
	if lver>=1: 
		size = min(lver,min(x.shape[0],x.shape[1])-1);
		w = gausWin(size,std)
		y = np.zeros((x.shape))
		# print "Gaussian window length: "+str(size)
		for i, row in enumerate(x):
			y[i] = np.convolve(row,w,mode='same')
	y = y.T.copy()
	if lhor>=1:
		size = min(lhor,min(x.shape[0],x.shape[1])-1);
		w = gausWin(size,std)
		# print "Gaussian window shape: " +str(w.shape)
		for i, col in enumerate(y):
			y[i] = np.convolve(col,w,mode='same')
	y = y.T.copy()
	return y

def gausWin(size,std):

	y = np.zeros(size)
	N = float((size-1))/float(2)
	for i in range(size):
		aux = (i-N)/(std*N)
		y[i] = np.exp(-0.5*aux*aux)
	return y

def storeAsArray(x):
	y = x[0]
	for i, row in enumerate(x,start=0):
		if i>0:
			y = np.append(y,row[:-i])
	return y

def unpackZeros(x,n): #unpacks the array padding the matrix with zeros on the right
	y = np.zeros((n,n))
	new_a = np.zeros(n)
	for i in range(n):
		idx1 = np.array(range(0,n-i))
		a = x[idx1]
		new_a = np.hstack((a,[0]*(n - len(a))))
		y[i] = new_a
		x = np.delete(x,idx1)
	# print y.shape
	return y

def shiftRows(x):
	y = np.zeros((x.shape))
	for i, row in enumerate(x,start=1):
		x = np.roll(row,i)
		y[i-1] = x
	return y

def convertToMatrix(x,n): # This copies the info	 of the left of the diagonal into the right.
	y = unpackZeros(x,n)
	y2 = np.vstack((y[1:],np.zeros(n)))
	y2 = shiftRows(y2)
	y2 = np.flipud(y2)
	y = y+y2
	return y

def extractSF(hpcps):
	y  = delayCoord(hpcps, m, tau)
	R  = ssm(y, kappa)
	L  = circShift(R)
	P  = gaussianBlur(L, lhor, lver, std=0.4)
	D  = downsample(P, 50)
	sf = storeAsArray(D)
	return sf

def downsample(x,n):
	d = signal.resample(x,n)
	d = signal.resample(d.T,n)
	return d.T

# @do_cprofile
def annotate(filename):

	print("file name: " + desc_path + filename)	
	hpcps = np.loadtxt(desc_path + filename, delimiter=',') # read descriptors

	plt.imshow(hpcps.T,interpolation='nearest',aspect='auto')
	plt.figure()

	y = delayCoord(hpcps, m, tau) # apply delay coordinates
	R = ssm(y,kappa)

	plt.imshow(R,cmap = cm.binary,interpolation='nearest')
	plt.figure()

	L = circShift(R)
	plt.subplot(211)
	plt.imshow(L,cmap = cm.binary,interpolation='nearest',aspect='auto')

	P = gaussianBlur(L,lhor,lver,std=0.4)

	plt.subplot(212)
	plt.imshow(P,cmap = cm.binary,interpolation='nearest',aspect='auto')
	plt.figure()

	print "P shape: " + str(P.shape)

	D = downsample(P,n)

	plt.imshow(D,cmap = cm.binary,interpolation='nearest',aspect='auto')
	plt.figure()

	arr = storeAsArray(D)
	unpacked = convertToMatrix(arr,n)

	plt.imshow(unpacked,cmap = cm.binary,interpolation='nearest',aspect='auto')
	plt.show()


if __name__ == "__main__":

	desc_path = '../hpcp_ah6_al5_csv/'
	res_path  = '../results'
	desc_list = 'desc_list.txt'
	filename  = 'Beatles_AllYouNeedIsLove_Beatles_1967-MagicalMysteryTour-11.wav.csv'

	annotate(filename)
