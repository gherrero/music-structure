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
import pickle

#-------------PARAMETERS-----------------#
hop_size    = 0.1393
min_len     = 3
m           = 22
kappa       = 0.03
lver        = 215
lhor        = 2
sigmaFactor = 1
tau         = 1
delta       = 0.1
lambd       = 5
n           = 100
std         = 0.4

np.set_printoptions(threshold=np.nan)
res_path = 'sfs/sf-alldatasets-n100/'

# -------PROFILING SCRIPTS---------#
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

#------STRUCTURE FEATURES EXTRACTION--------#

def delayCoord(x,m,tau):
	y = np.zeros((x.shape[0]-(m-1)*tau,x.shape[1]*m))
	for j in range(0,m):
		posj = j*x.shape[1]
		posi = j*tau
		y[:,posj:posj+x.shape[1]] = x[posi:posi+y.shape[0],:]
	return y

def ssm(X):
	K = int(kappa*X.shape[0])
	R = np.zeros([X.shape[0],X.shape[0]])
	i = 0
	tree=neighbors.KDTree(X,leaf_size=100,p=2.0)
	for i in range(X.shape[0]):
		index = tree.query(X[i,:],k=K,return_distance=False)
		R[i,index] = 1
	RT = R.T
	R = R*RT
	return R

def circShift(x):
	y = np.zeros((x.shape))
	for i, row in enumerate(x):
		x = np.roll(row,-i)
		y[i] = x
	return y.T

def gausWin(size):

	y = np.zeros(size)
	N = float((size-1))/float(2)
	for i in range(size):
		aux = (i-N)/(std*N)
		y[i] = np.exp(-0.5*aux*aux)
	return y

def gaussianBlur(x):
	if lver>=1: 
		size = min(lver,min(x.shape[0],x.shape[1])-1);
		w = gausWin(size)
		y = np.zeros((x.shape))
		for i, row in enumerate(x):
			y[i] = np.convolve(row,w,mode='same')
	y = y.T.copy()
	if lhor>=1:
		size = min(lhor,min(x.shape[0],x.shape[1])-1);
		w = gausWin(size)
		for i, col in enumerate(y):
			y[i] = np.convolve(col,w,mode='same')
	y = y.T.copy()
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
	return y

def shiftRows(x):
	y = np.zeros((x.shape))
	for i, row in enumerate(x,start=1):
		x = np.roll(row,i)
		y[i-1] = x
	return y

def convertToMatrix(x,n=n): # This copies the info on the left of the diagonal into the right side.
	y = unpackZeros(x,n)
	y2 = np.vstack((y[1:],np.zeros(n)))
	y2 = shiftRows(y2)
	y2 = np.flipud(y2)
	y = y+y2
	return y

def downsample(x):
	d = signal.resample(x,n)
	d = signal.resample(d.T,n)
	return d.T

def novelty(x):
	x=x.T
	c=[]
	for i in np.arange(len(x))-1:
		c.append(np.sqrt(sum((x[i]-x[i+1])**2)))
	# print "I still need to try to fix a way to get rid of spikes at the start and end"
	c=c[1:]
	minc=min(c)
	c-=minc
	maxc=max(c)
	c=c/maxc
	return c

def peakdet(x, section_length=lambd, threshold=delta):
	isMax = False
	peak_locations = []
	for i in np.arange(len(x)):
		if x[i]>=threshold:
			isMax = True
		for j in np.arange(1,section_length):
			if (i-j)>=0:
				if x[i-j]>x[i]:
					isMax = False
			if (i+j)<len(x):
				if x[i+j]>x[i]:
					isMax = False
		if isMax:
			peak_locations.append(i)
			i += section_length-1
	return peak_locations

def local_maxima(section):
    all_maxima = (section >= np.roll(section,  1, 0)) & (section >= np.roll(section, -1, 0)) & (section >= delta)
    return np.where(all_maxima)

def extractSF(hpcps):
	y  = delayCoord(hpcps, m, tau)
	R  = ssm(y)
	L  = circShift(R)
	P  = gaussianBlur(L)
	D  = downsample(P)
	sf = storeAsArray(D)
	return sf	

#-------PLOTTING AND PRINTING-------#
# @do_cprofile
def printProcess(filename):
	print("file name: " + desc_path + filename)	
	hpcps = np.loadtxt(desc_path + filename, delimiter=',') # read descriptors
	plt.imshow(hpcps.T,interpolation='nearest',aspect='auto')
	plt.figure()
	y = delayCoord(hpcps,m,tau) # apply delay coordinates
	R = ssm(y)
	plt.imshow(R,cmap = cm.binary,interpolation='nearest')
	# plt.title(filename)
	plt.figure()
	L = circShift(R)
	plt.subplot(211)
	plt.imshow(L,cmap = cm.binary,interpolation='nearest',aspect='auto')
	P = gaussianBlur(L)
	plt.figure()
	plt.subplot(212)
	plt.imshow(P,cmap = cm.binary,interpolation='nearest',aspect='auto')
	plt.figure()
	print "P shape: " + str(P.shape)
	D = downsample(P)
	plt.imshow(D,cmap = cm.binary,interpolation='nearest',aspect='auto')
	plt.figure()
	arr = storeAsArray(D)
	unpacked = convertToMatrix(arr,n)
	plt.imshow(unpacked,cmap = cm.binary,interpolation='nearest',aspect='auto')
	plt.figure('novelty')
	C = novelty(P)
	plt.plot(C)
	plt.title(filename)
	plt.show()

#--------BATCH PROCESSING----------#
def processFiles(filename):
	flist = open(filename,'r')
	for i,line in enumerate(flist,start=1):
		print str(i)+": "+line[:-1]
		f = open(res_path+line[:-1],'w')
		hpcps = np.loadtxt(desc_path + line[:-1], delimiter=',')
		sf = extractSF(hpcps)
		sf.tofile(f, sep="\n", format="%f")
		f.close()
	flist.close()

def storeAsCSVMatrix(filename):
	flist = open(filename,'r')
	for i,line in enumerate(flist,start=1):
		print str(i)+": "+line[:-1]
		hpcps = np.loadtxt(desc_path + line[:-1], delimiter=',')
		sf = extractSF(hpcps)
		np.savetxt(res_path+line[:-1],sf,delimiter='\t',fmt='%.5f')
	flist.close()

def storePickle(filename):
	flist = open(filename,'r')
	pick = open(pickle_fn,'w')
	for i, line in enumerate(flist,start=1):
		f = open(res_path+line[:-1],'r')
		data = np.loadtxt(res_path + line[:-1])
		print i
		pickle.dump(data,pick)
		f.close()
	flist.close()
	pick.close()

def getPickle(pickle_fn,list_fn):
	data = []
	pick = open(pickle_fn,'r')
	flist = open(list_fn,'r')
	for i, line in enumerate(flist,start=0):
		data.append(pickle.load(pick))
	pick.close()
	return data

def listToData(list_fn):
	print "extracting data from candidate list"
	flist = open(list_fn)
	lines = flist.readlines()
	data = []
	for i, line in enumerate(lines):
		data.append(np.loadtxt(res_path+line[:-4]+'csv'))
	return data

if __name__ == "__main__":

	desc_path = 'hpcp_ah6_al5_csv/'
	res_path  = 'sfs/sf-alldatasets-n100/'
	desc_list = 'sfs/alldatasets-n100.txt'
	pickle_fn = 'alldatasets-n1000.pickle'

	filename  = 'Beatles_AllYouNeedIsLove_Beatles_1967-MagicalMysteryTour-11.wav.csv'

	# printProcess(filename)
	# processFiles(desc_list)
	storePickle(desc_list)
	# printProcess(filename)
