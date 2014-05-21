import numpy as np
import cProfile
from sklearn import neighbors
import os.path
import structure_annotation as sa

gt_path    = 'metadata/all/'

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
			# print line
			# print ann_query
			duration_query = ann_query[0][-1][-2]

			ann_list = getAnnotationList(songs)
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
		ann_list = getAnnotationList(songs)


if __name__ == "__main__":

	desc_path  = 'hpcp_ah6_al5_csv/'
	gt_path    = 'metadata/all/'
	sf_path    = 'sfs/sf-alldatasets-n100/'

	sf_pickle  = 'pickles/sf-alldatasets-n100.pickle' #the one that includes the candidate list
	query_list = 'annotation_results/ann-chopin.txt'
	cand_list  = 'annotation_results/ann-alldatasets.txt'
	res_path   = 'annotation_results/chopin-n100/'

	K          = 10

	songList = open(cand_list).readlines()

	# storeResults(query_list,cand_list,res_path)
	printInfo(query_list,cand_list)

