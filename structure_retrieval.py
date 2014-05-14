import numpy as np
import cProfile
from sklearn import neighbors
import os.path
import structure_annotation as sa


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
		# print len(results_list)
		fn=gt_path+res[:-3]+"lab"
		if os.path.isfile(fn):
			f=open(fn,'r')
			lines=f.readlines()
			ann=[]
			for i, l in enumerate(lines):
				ann.append(l[:-1].split('\t'))
			ann_list.append(ann)
			f.close()
		else: print fn+" not found"
	return ann_list

def storeResults(list_fn,cand_list,res_path):
	f=open(list_fn,'r')
	filelist=f.readlines()
	train_set = sa.getPickle(sf_pickle,cand_list)
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
			songs = [songList[i][:-1] for i in songnumb if not songList[i][:-4]==line[:-4]]
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

def printInfo(list_fn,cand_list):
	f=open(list_fn)
	lines=f.readlines()
	train_set = sa.getPickle(sf_pickle,cand_list)
	tree = neighbors.KDTree(train_set,leaf_size=100,p=2.0)
	for i, line in enumerate(lines):
		fn=sf_path+line[:-4]+"csv"
		print "Query: "+line[:-5]
		sf_query = np.loadtxt(fn)
		songnumb = querySong(sf_query,tree).tolist()[0]
		songs = [songList[i][:-1] for i in songnumb if not songList[i][:-4]==line[:-4]]
		for song in songs: print "Result: "+song
		print "\n"
		ann_list = getAnnotationList(songs)

if __name__ == "__main__":

	filename  = 'Chopin_Op024No3_Bacha-1998_pid9166e-09.mp3.csv'

	desc_path  = 'hpcp_ah6_al5_csv/'
	sf_pickle  = 'pickles/alldatasets-n100.pickle'
	sf_path    = 'sfs/sf-alldatasets-n100/'
	query_list = 'annotation_results/ann-rwcIRISA-n100.txt'
	gt_path    = 'metadata/all/'
	res_path   = 'annotation_results/rwcIRISA-n100/'
	cand_list  = 'sfs/alldatasets-n100.txt'
	K          = 3

	songList = open(cand_list).readlines()

	storeResults(query_list,cand_list,res_path)
	# printInfo(query_list,cand_list)
