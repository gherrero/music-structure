import numpy as np
import cProfile
from sklearn import neighbors

import struct_annotation as sa
import structure_features as sf



def loadSongNames(filename):
	flist = open(filename,'r')
	songList = flist.readlines()
	return songList

def querySong(query,dataset):
	tree = neighbors.KDTree(dataset,leaf_size=100,p=2.0)
	index = tree.query(query,k=K,return_distance=False)
	return index

def unpickleAsArray(sf_pickle):
	dataM = []
	data = sf.getPickle(sf_pickle)
	for row in data:
		dataM.append(sa.convertToMatrix(row,50))
	return dataM

if __name__ == "__main__":

	desc_path = '../hpcp_ah6_al5_csv/'
	sf_pickle = 'dataset1.pickle'
	sf_path   = 'sf-dataset1/'
	list_fn   = 'dataset1.txt'
	filename  = 'Chopin_Op024No3_Bacha-1998_pid9166e-09.mp3.csv'

	K = 5
	songList = open(list_fn).readlines()

	print "Querying file: %s ..." %filename
	print "-"*100+"\n"
	hpcps_query = np.loadtxt(desc_path + filename, delimiter=',')

	print "HPCPS length: %s" %len(hpcps_query)
	print "Extracting structure features..."
	sf_query = sa.extractSF(hpcps_query)

	# print "Structure Features length: %s" %len(sf_query)
	print "..."
	print "Unpickling files from %s ..." %sf_pickle
	data = sf.getPickle(sf_pickle,list_fn)

	print "Total number of songs: %s" %len(data)
	songnumb = querySong(sf_query,data).tolist()[0] # [0] because .tolist() returns a nested list
	songs = [songList[i][:-1] for i in songnumb]

	print "\n"
	print "Query results:"
	print "--------------"
	for i, song in enumerate(songs): print str(songnumb[i]+1)+":\t"+str(song) #prints the number on the list not the index
	print "\n"

