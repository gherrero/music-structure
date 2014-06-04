import numpy as np
import cProfile
from sklearn import neighbors
import os.path
import structure_annotation as sa

gt_path   = 'metadata/all/'
cand_list = 'annotation_results/alldatasets.txt'
sf_pickle = 'pickles/sf-alldatasets-n100.pickle'
desc_path = 'hpcp_ah6_al5_csv/'
sf_path   = 'sfs/sf-alldatasets-n100/'
K         = 5

songList = open(cand_list).readlines()

def querySong(query,tree):
	index = tree.query(query,k=K,return_distance=False)
	return index

def getNeighbors(query_fn):
	sf_query = np.loadtxt(sf_path+query_fn)
	train_set = sa.getPickle(sf_pickle,cand_list)
	tree = neighbors.KDTree(train_set,leaf_size=100,p=2)
	songnumb = querySong(sf_query,tree).tolist()[0]
	neighbors_fn_list = [songList[i][:-1] for i in songnumb if not songList[i][:-5]==query_fn[:-4]]
	return neighbors_fn_list

def unpickleAsArray(sf_pickle):
	dataM = []
	data = sa.getPickle(sf_pickle)
	for row in data:
		dataM.append(sa.convertToMatrix(row,50))
	return dataM

def getAnnotationList(path,results_list):
	ann_list=[]
	for res in results_list:
		# print len(results_list)
		fn=path+res[:-3]+"lab"
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
	tree = neighbors.KDTree(train_set,leaf_size=100,p=2)

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
			ann_query = getAnnotationList(gt_path,[line[:-1]])
			duration_query = ann_query[0][-1][-2]

			ann_list = getAnnotationList(gt_path,songs)
			# print ann_list[0]
			duration_result=ann_list[0][-1][-2]

			r=float(duration_query)/float(duration_result)
			resFile=open(res_path+line[:-4]+'lab','w')
			for elem in ann_list[0]:
				label=elem[-1] 							 	# save the label so it's not modified
				elem = [float(el)*r for el in elem[:-1]] 	# rescale the boundaries according to duration
				elem.append(label) 						 	# append label again
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
		ann_list = getAnnotationList(gt_path,songs)

def listSelection(case):

	return {
		# (pickle, query, cand, results)
		1: ('pickles/sf-beatles-n100.pickle','annotation_results/beatlesQMUL.txt','annotation_results/beatlesQMUL.txt','annotation_results/beatlesQMUL-n100/'),
		2: ('pickles/sf-alldatasets-n100.pickle','annotation_results/beatlesQMUL.txt','annotation_results/alldatasets.txt','annotation_results/beatlesQMUL-n100/'),
		3: ('pickles/sf-chopin-n100.pickle','annotation_results/chopin.txt','annotation_results/chopin.txt','annotation_results/chopin-n100/'),
		4: ('pickles/sf-alldatasets-n100.pickle','annotation_results/chopin.txt','annotation_results/alldatasets.txt','annotation_results/chopin-n100/'),
		5: ('pickles/sf-rwcP-n100.pickle','annotation_results/rwcP.txt','annotation_results/rwcP.txt','annotation_results/rwcAIST-n100/'),
		6: ('pickles/sf-alldatasets-n100.pickle','annotation_results/rwcP.txt','annotation_results/alldatasets.txt','annotation_results/rwcAIST-n100/'),
	}[case]

if __name__ == "__main__":

	desc_path  = 'hpcp_ah6_al5_csv/'
	gt_path    = 'metadata/all/'
	sf_path    = 'sfs/sf-alldatasets-n100/'

	#	Case 1: beatlesQMUL vs beatlesQMUl
	#	Case 2: beatlesQMUL vs all
	#	Case 3: mazurkas vs mazurkas
	#	Case 4: mazurkas vs all
	#	Case 5: rwcP vs rwcP
	#	Case 6: rwcP vs all
	
	(sf_pickle, query_list,	cand_list, res_path) = listSelection(4)
	# songList = open(cand_list).readlines()

	# storeResults(query_list,cand_list,res_path)
	printInfo(query_list,cand_list)
	# n = getNeighbors('Chopin_Op006No1_Ashkenazy-1981_pid9058-01.mp3.csv')
	# print n
	# print len(n)


