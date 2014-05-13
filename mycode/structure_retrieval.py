import numpy as np
import cProfile
from sklearn import neighbors
import os.path
import struct_annotation as sa


def querySong(query,tree):
	index = tree.query(query,k=K,return_distance=False)
	return index

def unpickleAsArray(sf_pickle):
	dataM = []
	data = sa.getPickle(sf_pickle)
	for row in data:
		dataM.append(sa.convertToMatrix(row,50))
	return dataM

def getAnnotationList(results_list):
	ann_list=[]
	for res in results_list:
		fn=ann_path+res[:-3]+"lab"
		if os.path.isfile(fn):
			f=open(fn,'r')
			lines=f.readlines()
			ann=[]
			for i, l in enumerate(lines):
				ann.append(l[:-1].split('\t'))
			ann_list.append(ann)
			f.close()
	return ann_list

def storeResults(list_fn,res_path):
	f=open(list_fn,'r')
	filelist=f.readlines()
	train_set = sa.getPickle(sf_pickle,list_fn)
	tree = neighbors.KDTree(train_set,leaf_size=100,p=2.0)

	for i,line in enumerate(filelist):
		if i%50==0: print i
		# using directly the stored SF instead of extracting them each time.
		# hpcps_query = np.loadtxt(desc_path+line[:-1], delimiter=',')
		# sf_query = sa.extractSF(hpcps_query)
		fn=sf_path+line[:-4]+"csv"
		if os.path.isfile(fn):
			sf_query = np.loadtxt(fn)
			songnumb = querySong(sf_query,tree).tolist()[0]
			songs = [songList[i][:-1] for i in songnumb if not songList[i][:-1]==line[:-1]]
			# use duration information from query annotation file for now.
			ann_query = getAnnotationList([line[:-1]])
			duration_query = ann_query[0][-1][-2]
			ann_list = getAnnotationList(songs)
			duration_result=ann_list[0][-1][-2]

			r=float(duration_query)/float(duration_result)
			resFile=open(res_path+line[:-4]+'lab','w')
			for elem in ann_list[0]:
				elem = [float(el)*r for el in elem]  # this only works when labels are zeros, otherwise it messes everything up
				lin = ''.join([str(elem[j])+'\t' for j in range(len(elem))])[:-1]+'\n'
				resFile.writelines(lin)
			resFile.close()
		else:
			print fn+" not found"
	f.close()

def printInfo(list_fn):
	f=open(list_fn)
	lines=f.readlines()
	train_set = sa.getPickle(sf_pickle,list_fn)
	tree = neighbors.KDTree(train_set,leaf_size=100,p=2.0)
	for i, line in enumerate(lines):
		fn=sf_path+line[:-4]+"csv"
		print "Query: "+line[:-1]
		sf_query = np.loadtxt(fn)
		songnumb = querySong(sf_query,tree).tolist()[0]
		songs = [songList[i][:-1] for i in songnumb if not songList[i][:-1]==line[:-1]]
		print "Result: "+songs[0]
		print "\n"
		ann_list = getAnnotationList(songs)

if __name__ == "__main__":

	desc_path = '../hpcp_ah6_al5_csv/'
	sf_pickle = 'sf-beatles-n100.pickle'
	sf_path   = 'sf-beatles-n100/'
	list_fn   = 'ann-beatlesTUT-n100.txt'
	filename  = 'Chopin_Op024No3_Bacha-1998_pid9166e-09.mp3.csv'
	ann_path  = '../metadata/beatles/mylabfilesTUT_tab/'
	res_path  = 'annotation-results/beatlesTUT-n100/'
	K = 3

	songList = open(list_fn).readlines()

	# print "Querying file: %s ..." %filename
	# print "-"*100+"\n"
	# hpcps_query = np.loadtxt(desc_path + filename, delimiter=',')

	# print "HPCPS length: %s" %len(hpcps_query)
	# print "Extracting structure features..."
	# sf_query = sa.extractSF(hpcps_query)

	# print "Structure Features length: %s" %len(sf_query)
	# print "..."
	# print "Unpickling files from %s ..." %sf_pickle
	# data = sa.getPickle(sf_pickle,list_fn)
	# tree = neighbors.KDTree(data,leaf_size=100,p=2.0)


	# print "Total number of songs: %s" %len(data)
	# songnumb = querySong(sf_query,tree).tolist()[0] # [0] because .tolist() returns a nested list
	# songs = [songList[i][:-1] for i in songnumb if not songList[i][:-1]==filename] # removes query from results list

	# print "\n"
	# print "Query results:"
	# print "--------------"
	# for i, song in enumerate(songs): 
	# 	print str(i+1)+"-"+"\t"+str(songnumb[i]+1)+":\t"+str(song) #prints the NUMBER on the list not the index
	# print "\n"

	# ann_list = getAnnotationList(songs)

	storeResults(list_fn,res_path)
	# printInfo(list_fn)

