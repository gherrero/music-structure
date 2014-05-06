import struct_annotation as sa
import numpy as np
import pickle

def processFiles(filename):
	# print "Processing files from %s. Results stored in %s" % (filename,res_path)
	# print "----------------------------------------------------------------------"
	flist = open(filename,'r')
	for i,line in enumerate(flist,start=1):
		print str(i)+": "+line[:-1]
		f = open(res_path+line[:-1],'w')
		hpcps = np.loadtxt(desc_path + line[:-1], delimiter=',')
		sf = sa.extractSF(hpcps)
		sf.tofile(f, sep="\n", format="%f")
		f.close()
	flist.close()

def storePickle(filename):
	# print "Pickling files from %s into %ssf.pickle" % (filename,res_path)
	# print "----------------------------------------------------------------------"
	flist = open(filename,'r')
	pick = open(pickle_fn,'w')
	for i, line in enumerate(flist,start=1):
		f = open(res_path+line[:-1],'r')
		data = np.loadtxt(res_path + line[:-1], delimiter='\n')
		print i
		pickle.dump(data,pick)
		f.close()
	flist.close()
	pick.close()
		
def getPickle(pickle_fn,list_fn):
	# print "Unpickling files from %s ..." %pickle_fn
	data = []
	pick = open(pickle_fn,'r')
	flist = open(list_fn,'r')
	for i, line in enumerate(flist,start=0):
		data.append(pickle.load(pick))
		# print data
	pick.close()
	return data

if __name__ == "__main__":

	# COMENTADO PARA NO LIARLA AL EJECUTAR SIN QUERER

	desc_path = '../hpcp_ah6_al5_csv/'
	res_path  = 'sf-alldatasets/'
	list_fn   = 'desc_list.txt'
	pickle_fn = 'alldatasets.pickle'

	processFiles(list_fn)
	storePickle(list_fn)
	# data=getPickle(pickle_fn)

